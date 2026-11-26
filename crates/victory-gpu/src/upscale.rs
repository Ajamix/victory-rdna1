//! A plain upscaler in place of the title's AMD FidelityFX one, with
//! `VICTORY_UPSCALER=plain`: the four functions the title imports from
//! `amd_fidelityfx_loader_xbox.dll` (`victory-hle` serves them).
//!
//! The Series S title upscales 1280x720 to 1920x1080 with FSR 4, whose
//! console DLL holds only that one fixed-resolution model and refuses any
//! other render size. Here: no size checks, one jitter phase at (0, 0) (no
//! temporal accumulation to feed), and each dispatch one bilinear compute
//! pass from the colour target's render rectangle to the output, recorded
//! on the title's list. Exposure, reactive masks, sharpening: ignored.

use std::ffi::c_void;

const RETURN_OK: u32 = 0;
const CREATE_UPSCALE: u64 = 0x0001_0000;
const DISPATCH_UPSCALE: u64 = 0x0001_0001;
const QUERY_RATIO: u64 = 0x0001_0002;
const QUERY_RENDER_SIZE: u64 = 0x0001_0003;
const QUERY_PHASES: u64 = 0x0001_0004;
const QUERY_JITTER: u64 = 0x0001_0005;

/// The FFX API's description header: a type and the next description.
#[repr(C)]
pub struct Header {
    kind: u64,
    next: *const Header,
}

/// Whether the title's upscaler is replaced (`VICTORY_UPSCALER=plain`).
pub fn wanted() -> bool {
    std::env::var("VICTORY_UPSCALER").is_ok_and(|v| v.trim().eq_ignore_ascii_case("plain"))
}

/// The description of `kind` in the chain at `header`, or null.
fn find(mut header: *const Header, kind: u64) -> *const u8 {
    // SAFETY: the title passes a chain of live descriptions.
    while let Some(h) = unsafe { header.as_ref() } {
        if h.kind == kind {
            return header.cast();
        }
        header = h.next;
    }
    std::ptr::null()
}

fn read<T: Copy>(desc: *const u8, offset: usize) -> T {
    // SAFETY: `desc` is a live description at least `offset + size` long.
    unsafe { desc.add(offset).cast::<T>().read_unaligned() }
}

/// Logs the first sight of each description type (what the title asks for).
fn note(call: &str, header: *const Header) {
    static SEEN: std::sync::Mutex<Vec<(String, u64)>> = std::sync::Mutex::new(Vec::new());
    let mut seen = SEEN.lock().unwrap_or_else(|e| e.into_inner());
    let mut h = header;
    // SAFETY: as in `find`.
    while let Some(d) = unsafe { h.as_ref() } {
        if !seen.iter().any(|(c, k)| c == call && *k == d.kind) {
            seen.push((call.to_string(), d.kind));
            tracing::info!(call, kind = format_args!("{:#x}", d.kind), "plain upscaler: first description of this type");
        }
        h = d.next;
    }
}

/// `ffxCreateContext(ffxContext*, header*, allocation callbacks*)`.
pub extern "system" fn ffx_create_context(context: *mut *mut c_void, desc: *const Header, _alloc: *const c_void) -> u32 {
    note("create", desc);
    let up = find(desc, CREATE_UPSCALE);
    if !up.is_null() {
        let (rw, rh, ow, oh) = (read::<u32>(up, 0x14), read::<u32>(up, 0x18), read::<u32>(up, 0x1c), read::<u32>(up, 0x20));
        tracing::info!(flags = format_args!("{:#x}", read::<u32>(up, 0x10)), max_render = format_args!("{rw}x{rh}"), max_output = format_args!("{ow}x{oh}"), "plain upscaler replaces FidelityFX (VICTORY_UPSCALER=plain)");
    }
    if !context.is_null() {
        // SAFETY: the title's context slot; any non-null value names a context.
        unsafe { context.write(Box::into_raw(Box::new(0u64)).cast()) };
    }
    RETURN_OK
}

/// `ffxDestroyContext(ffxContext*, allocation callbacks*)`.
/// shortcut: the 8-byte context is leaked; titles make one per resolution change.
pub extern "system" fn ffx_destroy_context(_context: *mut *mut c_void, _alloc: *const c_void) -> u32 {
    RETURN_OK
}

/// `ffxConfigure(ffxContext*, header*)`: nothing to configure.
pub extern "system" fn ffx_configure(_context: *mut *mut c_void, desc: *const Header) -> u32 {
    note("configure", desc);
    RETURN_OK
}

/// The scale FSR's quality modes divide the output by.
fn ratio(quality: u32) -> f32 {
    match quality {
        0 => 1.0,
        1 => 1.5,
        2 => 1.7,
        3 => 2.0,
        _ => 3.0,
    }
}

/// `ffxQuery(ffxContext*, header*)`.
pub extern "system" fn ffx_query(_context: *mut *mut c_void, desc: *mut Header) -> u32 {
    note("query", desc);
    // SAFETY: the title passes a live description; the out pointers it holds
    // are the title's, written as the FFX API defines them.
    unsafe {
        let Some(h) = desc.as_ref() else { return RETURN_OK };
        let d = desc.cast::<u8>().cast_const();
        match h.kind {
            QUERY_PHASES => {
                if let Some(out) = read::<*mut i32>(d, 0x18).as_mut() {
                    *out = 1;
                }
            }
            QUERY_JITTER => {
                for at in [0x18, 0x20] {
                    if let Some(out) = read::<*mut f32>(d, at).as_mut() {
                        *out = 0.0;
                    }
                }
            }
            QUERY_RENDER_SIZE => {
                let r = ratio(read::<u32>(d, 0x18));
                for (size, at) in [(read::<u32>(d, 0x10), 0x20), (read::<u32>(d, 0x14), 0x28)] {
                    if let Some(out) = read::<*mut u32>(d, at).as_mut() {
                        *out = (size as f32 / r) as u32;
                    }
                }
            }
            QUERY_RATIO => {
                if let Some(out) = read::<*mut f32>(d, 0x18).as_mut() {
                    *out = ratio(read::<u32>(d, 0x10));
                }
            }
            _ => {}
        }
    }
    RETURN_OK
}

/// `ffxDispatch(ffxContext*, header*)`: the upscale description is a header,
/// the command list (+0x10), six resources of 0x30 bytes each from +0x18
/// (colour, depth, motion vectors, exposure, reactive, transparency), the
/// output (+0x138), jitter, motion vector scale, then render size (+0x178)
/// and upscale size (+0x180).
pub extern "system" fn ffx_dispatch(_context: *mut *mut c_void, desc: *const Header) -> u32 {
    note("dispatch", desc);
    let d = find(desc, DISPATCH_UPSCALE);
    if d.is_null() {
        return RETURN_OK;
    }
    let list = read::<*mut c_void>(d, 0x10);
    let color = read::<*mut c_void>(d, 0x18);
    let output = read::<*mut c_void>(d, 0x138);
    let render = (read::<u32>(d, 0x178), read::<u32>(d, 0x17c));
    #[cfg(windows)]
    if record(list.cast_const().cast(), color.cast_const().cast(), output.cast_const().cast(), render).is_none() {
        static ONCE: std::sync::Once = std::sync::Once::new();
        ONCE.call_once(|| tracing::warn!("plain upscaler: a dispatch was not recorded (no host list, texture or pipeline)"));
    }
    #[cfg(not(windows))]
    let _ = (list, color, output, render);
    RETURN_OK
}

#[cfg(windows)]
const HLSL: &str = r#"
#define RS "RootConstants(num32BitConstants=4, b0), DescriptorTable(SRV(t0), UAV(u0)), StaticSampler(s0, filter=FILTER_MIN_MAG_MIP_LINEAR, addressU=TEXTURE_ADDRESS_CLAMP, addressV=TEXTURE_ADDRESS_CLAMP)"
cbuffer Scale : register(b0) { float2 uv_per_pixel; float2 size; };
Texture2D src : register(t0);
RWTexture2D<float4> dst : register(u0);
SamplerState smp : register(s0);
[RootSignature(RS)]
[numthreads(8, 8, 1)]
void upscale(uint3 id : SV_DispatchThreadID) {
    if (id.x >= (uint)size.x || id.y >= (uint)size.y) return;
    dst[id.xy] = src.SampleLevel(smp, (float2(id.xy) + 0.5) * uv_per_pixel, 0);
}
"#;

#[cfg(windows)]
const VIEWS: u32 = 256;

#[cfg(windows)]
struct Upscaler {
    root: windows::Win32::Graphics::Direct3D12::ID3D12RootSignature,
    pipeline: windows::Win32::Graphics::Direct3D12::ID3D12PipelineState,
    /// A ring of (SRV, UAV) pairs; frames in the air read theirs.
    views: windows::Win32::Graphics::Direct3D12::ID3D12DescriptorHeap,
    step: u32,
    next: u32,
}

// SAFETY: D3D12 objects are free-threaded; the ring is behind a mutex.
#[cfg(windows)]
unsafe impl Send for Upscaler {}

#[cfg(windows)]
fn upscaler() -> Option<&'static std::sync::Mutex<Upscaler>> {
    use windows::Win32::Graphics::Direct3D12::*;
    use windows::core::s;
    static UPSCALER: std::sync::OnceLock<Option<std::sync::Mutex<Upscaler>>> = std::sync::OnceLock::new();
    UPSCALER
        .get_or_init(|| {
            let make = || -> windows::core::Result<Upscaler> {
                let device = &crate::host::host().ok_or_else(windows::core::Error::empty)?.device;
                let cs = crate::host::compile(HLSL, s!("upscale"), s!("cs_5_1"))?;
                // SAFETY: the root signature is inside the shader; the
                // description points at live data.
                unsafe {
                    let root: ID3D12RootSignature = device.CreateRootSignature(0, &cs[..])?;
                    let pipeline = device.CreateComputePipelineState(&D3D12_COMPUTE_PIPELINE_STATE_DESC {
                        pRootSignature: std::mem::transmute_copy(&root),
                        CS: D3D12_SHADER_BYTECODE { pShaderBytecode: cs.as_ptr().cast(), BytecodeLength: cs.len() },
                        ..Default::default()
                    })?;
                    let views = device.CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                        Type: D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
                        NumDescriptors: VIEWS * 2,
                        Flags: D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE,
                        NodeMask: 0,
                    })?;
                    let step = device.GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
                    Ok(Upscaler { root, pipeline, views, step, next: 0 })
                }
            };
            make().map_err(|e| tracing::warn!(error = %e, "plain upscaler: no pipeline")).ok().map(std::sync::Mutex::new)
        })
        .as_ref()
}

/// Records the upscale of `color`'s top-left `render` rectangle to all of
/// `output` on the title's `list`, then has the title's next commands bind
/// their own heaps, roots and pipeline again.
#[cfg(windows)]
pub(crate) fn record(list: crate::object::This, color: crate::object::This, output: crate::object::This, render: (u32, u32)) -> Option<()> {
    use windows::Win32::Graphics::Direct3D12::*;
    use windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT;
    const SHADER_READ: u32 = 0x40; // NON_PIXEL_SHADER_RESOURCE
    const SHADER_WRITE: u32 = 0x8; // UNORDERED_ACCESS
    let host = crate::list::host_list(list)?;
    let src = crate::resource::host_resource(color)?;
    let dst = crate::resource::host_resource(output)?;
    crate::list::ensure_state(host, color, SHADER_READ);
    crate::list::ensure_state(host, output, SHADER_WRITE);
    let device = &crate::host::host()?.device;
    let mut up = upscaler()?.lock().unwrap_or_else(|e| e.into_inner());
    // SAFETY: live textures and heap; an open list; the views are written
    // before the list that reads them is submitted.
    unsafe {
        let (sd, dd) = (src.GetDesc(), dst.GetDesc());
        let (rw, rh) = if render.0 == 0 || render.1 == 0 { (sd.Width as u32, sd.Height) } else { render };
        let slot = up.next % VIEWS;
        up.next = up.next.wrapping_add(1);
        let mut cpu = up.views.GetCPUDescriptorHandleForHeapStart();
        cpu.ptr += (slot * 2 * up.step) as usize;
        let mut gpu = up.views.GetGPUDescriptorHandleForHeapStart();
        gpu.ptr += u64::from(slot * 2 * up.step);
        let view = |f: DXGI_FORMAT| crate::host::plane_view_format(f);
        device.CreateShaderResourceView(
            src,
            Some(&D3D12_SHADER_RESOURCE_VIEW_DESC {
                Format: view(sd.Format),
                ViewDimension: D3D12_SRV_DIMENSION_TEXTURE2D,
                Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
                Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 { Texture2D: D3D12_TEX2D_SRV { MipLevels: 1, ..Default::default() } },
            }),
            cpu,
        );
        let uav = D3D12_CPU_DESCRIPTOR_HANDLE { ptr: cpu.ptr + up.step as usize };
        device.CreateUnorderedAccessView(
            dst,
            None,
            Some(&D3D12_UNORDERED_ACCESS_VIEW_DESC {
                Format: view(dd.Format),
                ViewDimension: D3D12_UAV_DIMENSION_TEXTURE2D,
                Anonymous: D3D12_UNORDERED_ACCESS_VIEW_DESC_0 { Texture2D: D3D12_TEX2D_UAV { MipSlice: 0, PlaneSlice: 0 } },
            }),
            uav,
        );
        let (ow, oh) = (dd.Width as f32, dd.Height as f32);
        let scale: [f32; 4] = [rw as f32 / sd.Width as f32 / ow, rh as f32 / sd.Height as f32 / oh, ow, oh];
        let l = &host.list;
        l.SetDescriptorHeaps(&[Some(up.views.clone())]);
        l.SetComputeRootSignature(&up.root);
        l.SetPipelineState(&up.pipeline);
        l.SetComputeRoot32BitConstants(0, 4, scale.as_ptr().cast(), 0);
        l.SetComputeRootDescriptorTable(1, gpu);
        l.Dispatch((dd.Width as u32).div_ceil(8), dd.Height.div_ceil(8), 1);
    }
    drop(up);
    crate::list::replay_state(list);
    Some(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The jitter answer the title divides by is one phase at (0, 0), and the
    /// render size of a quality mode is the output over its ratio.
    #[test]
    fn queries_answer_one_phase_and_the_ratio_sizes() {
        let mut phases = 0i32;
        let mut q = [0u8; 0x20];
        q[..8].copy_from_slice(&QUERY_PHASES.to_le_bytes());
        q[0x18..0x20].copy_from_slice(&(&mut phases as *mut i32 as u64).to_le_bytes());
        ffx_query(std::ptr::null_mut(), q.as_mut_ptr().cast());
        assert_eq!(phases, 1);

        let (mut w, mut h) = (0u32, 0u32);
        let mut q = [0u8; 0x30];
        q[..8].copy_from_slice(&QUERY_RENDER_SIZE.to_le_bytes());
        q[0x10..0x14].copy_from_slice(&1920u32.to_le_bytes());
        q[0x14..0x18].copy_from_slice(&1080u32.to_le_bytes());
        q[0x18..0x1c].copy_from_slice(&3u32.to_le_bytes());
        q[0x20..0x28].copy_from_slice(&(&mut w as *mut u32 as u64).to_le_bytes());
        q[0x28..0x30].copy_from_slice(&(&mut h as *mut u32 as u64).to_le_bytes());
        ffx_query(std::ptr::null_mut(), q.as_mut_ptr().cast());
        assert_eq!((w, h), (960, 540));
    }
}
