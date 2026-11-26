//! Native GFX10 programs that the console compiler stores in a `DXIL` part.
//!
//! An `XDAT` index after the program header lists blobs stored around the
//! bitcode; one or more of them hold native programs, zlib-packed (a vertex
//! shader can carry two: compiled for use with and without a geometry
//! shader). Layout as observed in shipped titles (index offsets are relative
//! to the part data):
//!
//! ```text
//! +0x18 "XDAT"  u32 entry count, u32 (0x24 seen), then per entry:
//!               u32 kind, u32 offset, u32 size
//! kind 0x0d     "XBCH" u32 version (3), u64 packed size, u64 unpacked size,
//!               zlib stream
//! unpacked      "FSSM" u32 version (0x101), u64 payload size, payload
//! payload +0x00 u64 graphics slot count (6)
//!         +0x08 u64 record offsets: LS HS ES GS VS PS (0x08..0x30),
//!               mesh 0x48, compute 0x60; unused ones are 0xDEADBEEF_DEADBEEF
//! record  +0x00 u64 code offset, +0x08 u64 code size (payload-relative),
//!         +0x14 u32 flags: bit 5 wave32 (else wave64)
//!         +0x28 u64 root layout table, +0x30 u64 used-resource table
//!               (payload-relative), +0x38 u32 and +0x3c u32 their counts
//!         +0x50 u32 bits 0-9 VGPRs used, bits 20-27 SGPRs used
//! layout  20 B: u32 root type, u32 table offset (dwords), u32 user SGPRs,
//!               u32 base register, u8 first user-data dword, u8 root slot,
//!               u16 0
//! used    20 B: u8 resource type, u8 root type, u16 0, u32 table offset
//!               (dwords), u32 lower bound, u32 stride, u8 descriptor
//!               dwords, u8 first user-data dword, u8 root slot, u8 0
//! ```
//!
//! Stage data follows the record; offsets from its end (E = record + 0x54):
//!
//! ```text
//! PS  E+0x24  32 inputs of 8 B: u8 usage, u8 signature row, u8 channel
//!             mask, u8 attribute (v_interp `attr`), u8 default, u8 second
//!             row (of a packed pair), u8 parameter slot, u8 flags (bit 0
//!             flat, 1 linear, 3 fp16, 4 fp16 packed); E+0x124 u32 count
//!     E+0x134 u32 SPI_PS_INPUT_ADDR, E+0x138 u32 DB_SHADER_CONTROL
//!     E+0x148 u32 export patch size (dwords), E+0x14c 8 patches, one per
//!             MRT: u32 code offset (dwords, -1 none), then the export code
//!             for each color format in 10 slots of 10 dwords (ZERO, 32_R,
//!             32_GR, 32_AR, FP16, UNORM16, SNORM16, UINT16, SINT16, 32_ABGR)
//! VS  E+0x24  32 inputs of 5 B: u8 usage, u8 row, u8 channel mask, u8 VGPR
//!             of channel 0 (channel k in VGPR + k), u8 highest channel;
//!             E+0xc4 u32 count
//!     E+0xc8  32 outputs of 6 B: u8 usage, u8 row, u8 parameter (`exp
//!             param<n>`), u8 row again, u8 slot, u8 flags (bit 0 fp16,
//!             bit 1 two fp16 values per dword); E+0x188 u32 count
//! CS  E+0x24  u32 COMPUTE_PGM_RSRC2, E+0x28 u32 x3 thread-group size
//! MS  E+0xc8  outputs as VS, E+0x188 count; the param byte's bit 7 marks a
//!             per-primitive output (`PSG1` row); the 3 bytes more are the
//!             second row of a packed pair, a slot, flags (bit 0 fp16, bit 1
//!             two fp16 elements packed)
//!     E+0x18c u32 SPI_VS_OUT_CONFIG, SPI_SHADER_POS_FORMAT,
//!             PA_CL_VS_OUT_CNTL, VGT_STRMOUT_CONFIG, SPI_SHADER_PGM_RSRC2_GS,
//!             VGT_GS_OUT_PRIM_TYPE, VGT_GS_INSTANCE_CNT, SPI_SHADER_IDX_FORMAT
//!     E+0x448 u16 thread-group size in lanes, u16 maximum vertices,
//!             u16 maximum primitives, u8 user SGPRs, u8 output topology
//!             (2 triangles); code follows at E+0x45c
//! record +0x4c u32 LDS bytes (`lds_byte_size`; mesh programs read it here)
//! ```
//!
//! Rows are registers of the container's signature parts (`ISG1`, `OSG1`).
//! Merged stages keep their stage data in another layout (not read).
//!
//! Other kinds seen: 0x07 program header + bitcode (or a stub when stripped),
//! 0x0a `xbop` (compiler options), 0x0e a table not decoded yet, 0x10 and
//! 0x12 16-byte blobs (0x10 is the shader hash).
//! Unknown versions are rejected so a format change shows up as an error.

use crate::ShaderError::{self, Malformed};
use crate::dxbc::{u32_at, u64_at};

const XDAT_OFFSET: usize = 0x18;
const MAX_XDAT_ENTRIES: usize = 64;
const KIND_NATIVE: u32 = 0x0d;
const XBCH_VERSION: u32 = 3;
const XBCH_HEADER: usize = 0x18;
const FSSM_VERSION: u32 = 0x101;
const FSSM_HEADER: usize = 0x10;
const GRAPHICS_SLOT_COUNT: u64 = 6;
/// Where each stage's record offset sits in the FSSM payload.
const SLOTS: [(HwStage, usize); 8] = [
    (HwStage::Ls, 0x08),
    (HwStage::Hs, 0x10),
    (HwStage::Es, 0x18),
    (HwStage::Gs, 0x20),
    (HwStage::Vs, 0x28),
    (HwStage::Ps, 0x30),
    (HwStage::Ms, 0x48),
    (HwStage::Cs, 0x60),
];
const ABSENT: u64 = 0xDEAD_BEEF_DEAD_BEEF;
const RECORD_SIZE: u64 = 0x54;
const RECORD_FLAGS: usize = 0x14;
const FLAG_WAVE32: u32 = 1 << 5;
const RECORD_REGS: usize = 0x50;
const RECORD_LAYOUT: usize = 0x28;
const RECORD_USED: usize = 0x30;
const RECORD_COUNTS: usize = 0x38;
const ENTRY_SIZE: u64 = 20;
const MAX_SEMANTICS: u32 = 32;
const PS_INPUTS: usize = 0x24;
const PS_INPUT_COUNT: usize = 0x124;
const PS_INPUT_ADDR: usize = 0x134;
const PS_DB_SHADER_CONTROL: usize = 0x138;
const PS_PATCH_SIZE: usize = 0x148;
const PS_PATCHES: usize = 0x14c;
const PATCH_SLOT_DWORDS: usize = 10;
/// The variant that exports every channel as a raw 32-bit value.
const PATCH_FULL: usize = 9;
const PATCH_BYTES: usize = 4 + 10 * PATCH_SLOT_DWORDS * 4;
const MRTS: usize = 8;
const VS_INPUTS: usize = 0x24;
const VS_INPUT_COUNT: usize = 0xc4;
const VS_OUTPUTS: usize = 0xc8;
const VS_OUTPUT_COUNT: usize = 0x188;
const CS_RSRC2: usize = 0x24;
const CS_THREADS: usize = 0x28;
const RECORD_LDS: usize = 0x4c;
const MS_CLIP_CNTL: usize = 0x194;
const MS_INFO: usize = 0x448;
/// Unpacked blobs are a few KiB; refuse to allocate for absurd sizes.
const MAX_UNPACKED: u64 = 16 << 20;

/// Hardware shader stage a native program runs as.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum HwStage {
    Ls,
    Hs,
    Es,
    Gs,
    Vs,
    Ps,
    /// Mesh shader.
    Ms,
    Cs,
}

impl HwStage {
    /// The SGPR holding user-data dword `index`: merged stages (and mesh
    /// shaders, which run as one) have 8 system SGPRs first.
    pub fn user_sgpr(self, index: u8) -> u8 {
        match self {
            HwStage::Ls | HwStage::Hs | HwStage::Es | HwStage::Gs | HwStage::Ms => index + 8,
            HwStage::Vs | HwStage::Ps | HwStage::Cs => index,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NativeProgram {
    pub stage: HwStage,
    /// Runs as 64 lanes per wave (else 32); sizes lane masks.
    pub wave64: bool,
    /// VGPRs the program uses (v0 up to this), the bound of indexed access.
    pub vgprs: u16,
    /// Where the root signature's parameters arrive (user SGPRs).
    pub layout: Vec<RootParam>,
    /// The descriptors the program reads, by root parameter.
    pub used: Vec<UsedResource>,
    pub io: StageIo,
    /// GFX10 machine code, as stored.
    pub code: Vec<u8>,
}

impl NativeProgram {
    /// The code with each render target's export patched for its full
    /// 32-bit format, so exports carry the shader's values unconverted (as
    /// the console does at pipeline creation for that format).
    pub fn patched_code(&self) -> Vec<u8> {
        let mut code = self.code.clone();
        if let StageIo::Ps { patches, .. } = &self.io {
            for p in patches {
                let at = p.offset as usize * 4;
                for (i, word) in p.code.iter().enumerate() {
                    code[at + 4 * i..at + 4 * i + 4].copy_from_slice(&word.to_le_bytes());
                }
            }
        }
        code
    }
}

/// What a semantic is, as the stage data names it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Usage {
    Generic,
    ClipDistance,
    Other(u8),
}

impl Usage {
    fn from_raw(raw: u8) -> Self {
        match raw {
            6 => Usage::Generic,
            7 => Usage::ClipDistance,
            n => Usage::Other(n),
        }
    }
}

/// A pixel shader input: what `v_interp` of attribute `attr` reads.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PsInput {
    pub usage: Usage,
    /// Register in the input signature.
    pub row: u8,
    pub mask: u8,
    pub attr: u8,
    pub flat: bool,
    pub fp16: bool,
    /// Two 16-bit values per dword ([`VsOutput::pack2`]); `row2` is the
    /// second element's register.
    pub pack2: bool,
    pub row2: u8,
}

/// A vertex attribute the fetch shader loads: channel k into VGPR `vgpr + k`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VsInput {
    pub usage: Usage,
    pub row: u8,
    pub mask: u8,
    pub vgpr: u8,
}

/// What `exp param<param>` of a vertex shader writes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VsOutput {
    pub usage: Usage,
    /// Register in the output signature.
    pub row: u8,
    pub param: u8,
    /// 16-bit values, one per dword (low half) unless `pack2`: then two per
    /// dword, channel c of this row in dword c / 2 (odd channels in the
    /// high half), then a second 16-bit row from the next whole dword (the
    /// next 16-bit row of the signature no entry names).
    pub fp16: bool,
    pub pack2: bool,
}

/// The export code of one render target for its full 32-bit format.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExportPatch {
    pub mrt: u8,
    /// Where it goes in the code, in dwords.
    pub offset: u32,
    pub code: Vec<u32>,
}

/// Inputs and outputs, for the stages whose stage data is read.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub enum StageIo {
    #[default]
    None,
    Vs {
        inputs: Vec<VsInput>,
        outputs: Vec<VsOutput>,
    },
    Ps {
        inputs: Vec<PsInput>,
        /// `SPI_PS_INPUT_ADDR`: which system values arrive in VGPRs.
        input_addr: u32,
        /// `DB_SHADER_CONTROL`: depth export, kill.
        db_shader_control: u32,
        patches: Vec<ExportPatch>,
    },
    Cs(ComputeIo),
    Ms(MeshIo),
}

/// A mesh program's outputs and limits. It runs as one hardware NGG stage:
/// lane n of the group (`threads` lanes) exports vertex n (`exp pos`,
/// `exp param`) and primitive n (`exp prim`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MeshIo {
    /// `exp param<n>` targets; `param` bit 7 set: a per-primitive output,
    /// `row` then a row of the primitive signature (`PSG1`).
    pub outputs: Vec<VsOutput>,
    /// `PA_CL_VS_OUT_CNTL`: which clip and cull distances `pos1+` carry.
    pub clip_cntl: u32,
    /// Thread-group size in lanes (the HLSL group rounded up to waves).
    pub threads: u32,
    pub max_vertices: u32,
    pub max_primitives: u32,
    /// 1 lines, 2 triangles.
    pub topology: u8,
    pub lds_bytes: u32,
}

/// A compute program's dispatch settings.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ComputeIo {
    /// `COMPUTE_PGM_RSRC2`: user SGPRs, which system SGPRs and VGPRs
    /// follow them, LDS size.
    pub rsrc2: u32,
    /// Thread-group size (`numthreads`).
    pub threads: [u32; 3],
}

impl ComputeIo {
    /// User SGPRs before the system ones.
    pub fn user_sgprs(&self) -> u8 {
        (self.rsrc2 >> 1 & 0x1F) as u8
    }

    /// The SGPR of each enabled thread-group id component, in order after
    /// the user SGPRs, then the thread-group info SGPR if enabled.
    pub fn system_sgprs(&self) -> ([Option<u8>; 3], Option<u8>) {
        let mut next = self.user_sgprs();
        let mut take = |bit: u32| {
            (self.rsrc2 >> bit & 1 != 0).then(|| {
                next += 1;
                next - 1
            })
        };
        let ids = [take(7), take(8), take(9)];
        (ids, take(10))
    }

    /// SGPRs the hardware initializes: the user SGPRs, the enabled system
    /// ones, then the scratch wave offset when scratch is enabled (bit 0).
    /// The ones above hold whatever the previous wave left.
    pub fn initialized_sgprs(&self) -> u8 {
        let (ids, info) = self.system_sgprs();
        let end = ids.into_iter().chain([info]).flatten().map(|s| s + 1).max();
        end.unwrap_or(0).max(self.user_sgprs()) + (self.rsrc2 & 1) as u8
    }

    /// Thread id components in v0, v1, v2 (1 to 3).
    pub fn thread_id_vgprs(&self) -> u8 {
        (self.rsrc2 >> 11 & 3) as u8 + 1
    }

    /// LDS bytes per thread group (512-byte granules).
    pub fn lds_bytes(&self) -> u32 {
        (self.rsrc2 >> 15 & 0x1FF) * 512
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RootType {
    Constants,
    /// A 64-bit buffer address.
    Descriptor,
    /// The 32-bit heap address of a descriptor table range.
    Table,
    /// A sampler the root signature fixes.
    ImmediateSampler,
    /// The 32-bit address of the CBV/SRV/UAV heap (SM 6.6 dynamic resources).
    ResourceHeap,
    /// The 32-bit address of the sampler heap.
    SamplerHeap,
    Other(u32),
}

impl RootType {
    fn from_raw(raw: u32) -> Self {
        match raw {
            0 => RootType::Constants,
            1 => RootType::Descriptor,
            3 => RootType::Table,
            4 => RootType::ImmediateSampler,
            5 => RootType::ResourceHeap,
            6 => RootType::SamplerHeap,
            n => RootType::Other(n),
        }
    }
}

/// One root parameter (or table range) and the user SGPRs holding it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RootParam {
    pub root: RootType,
    /// Of the range within its table.
    pub table_offset_dwords: u32,
    /// User SGPRs from `user_data` on.
    pub sgprs: u32,
    /// Shader register of the first descriptor (`b`/`t`/`u`/`s` number).
    pub base_register: u32,
    /// First user-data dword ([`HwStage::user_sgpr`]).
    pub user_data: u8,
    /// Index of the parameter in the root signature.
    pub slot: u8,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ResourceType {
    Srv,
    Sampler,
    Cbv,
    Uav,
    Other(u8),
}

/// A descriptor range the program reads.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct UsedResource {
    pub kind: ResourceType,
    pub root: RootType,
    pub table_offset_dwords: u32,
    /// Shader register of the first descriptor read.
    pub lower_bound: u32,
    pub stride: u32,
    /// Size of the descriptor (4 or 8).
    pub dwords: u8,
    pub user_data: u8,
    pub slot: u8,
}

/// The native programs in a `DXIL` part's data; empty for the PC layout.
pub fn native_programs(part: &[u8]) -> Result<Vec<NativeProgram>, ShaderError> {
    if part.get(XDAT_OFFSET..XDAT_OFFSET + 4) != Some(b"XDAT") {
        return Ok(Vec::new());
    }
    let count = u32_at(part, XDAT_OFFSET + 4)? as usize;
    if count > MAX_XDAT_ENTRIES {
        return Err(Malformed("implausible XDAT entry count"));
    }
    let mut out = Vec::new();
    for i in 0..count {
        let at = XDAT_OFFSET + 12 + 12 * i;
        if u32_at(part, at)? != KIND_NATIVE {
            continue;
        }
        let offset = u32_at(part, at + 4)?.into();
        let size = u32_at(part, at + 8)?.into();
        let blob = bytes(part, offset, size).ok_or(Malformed("XDAT entry out of bounds"))?;
        out.extend(programs(&unpack(blob)?)?);
    }
    Ok(out)
}

fn unpack(blob: &[u8]) -> Result<Vec<u8>, ShaderError> {
    if blob.get(0..4) != Some(b"XBCH") {
        return Err(Malformed("native blob lacks XBCH magic"));
    }
    if u32_at(blob, 4)? != XBCH_VERSION {
        return Err(Malformed("unknown XBCH version"));
    }
    let packed = u64_at(blob, 8)?;
    let unpacked = u64_at(blob, 16)?;
    if unpacked > MAX_UNPACKED {
        return Err(Malformed("implausible XBCH unpacked size"));
    }
    let stream =
        bytes(blob, XBCH_HEADER as u64, packed).ok_or(Malformed("XBCH stream out of bounds"))?;
    let out = inflate_zlib_exact(stream, unpacked as usize)
        .ok_or(Malformed("XBCH stream does not inflate to its stated size"))?;
    Ok(out)
}

/// Inflates a zlib stream incrementally up to `expected` bytes, rejecting
/// truncated, corrupt, under-filled, or oversized streams without eagerly
/// allocating the full declared size up front.
fn inflate_zlib_exact(source: &[u8], expected: usize) -> Option<Vec<u8>> {
    let mut state = zlib_rs::Inflate::new(true, 15);
    let mut out = Vec::new();
    let (mut in_pos, mut out_pos) = (0usize, 0usize);
    loop {
        if out_pos == expected {
            let mut extra = [0u8; 1];
            let status = state
                .decompress(
                    &source[in_pos..],
                    &mut extra,
                    zlib_rs::InflateFlush::NoFlush,
                )
                .ok()?;
            return (status == zlib_rs::Status::StreamEnd
                && state.total_out() as usize == expected)
                .then_some(out);
        }
        if out_pos == out.len() {
            let next_len = if out.is_empty() {
                expected.min(source.len().saturating_mul(4).max(4096))
            } else {
                out.len().saturating_mul(2).min(expected)
            };
            out.resize(next_len, 0);
        }
        let status = state
            .decompress(
                &source[in_pos..],
                &mut out[out_pos..],
                zlib_rs::InflateFlush::NoFlush,
            )
            .ok()?;
        let next_in = state.total_in() as usize;
        let next_out = state.total_out() as usize;
        match status {
            zlib_rs::Status::StreamEnd => {
                out.truncate(next_out);
                return (next_out == expected).then_some(out);
            }
            zlib_rs::Status::Ok if next_in != in_pos || next_out != out_pos => {
                in_pos = next_in;
                out_pos = next_out;
            }
            zlib_rs::Status::Ok | zlib_rs::Status::BufError => return None,
        }
    }
}

fn programs(fssm: &[u8]) -> Result<Vec<NativeProgram>, ShaderError> {
    if fssm.get(0..4) != Some(b"FSSM") {
        return Err(Malformed("native blob lacks FSSM magic"));
    }
    if u32_at(fssm, 4)? != FSSM_VERSION {
        return Err(Malformed("unknown FSSM version"));
    }
    let payload = &fssm[FSSM_HEADER.min(fssm.len())..];
    if u64_at(fssm, 8)? != payload.len() as u64 {
        return Err(Malformed("FSSM payload size disagrees"));
    }
    if u64_at(payload, 0)? != GRAPHICS_SLOT_COUNT {
        return Err(Malformed("unexpected FSSM slot count"));
    }
    let mut out = Vec::new();
    for (stage, at) in SLOTS {
        let start = u64_at(payload, at)?;
        if start == ABSENT {
            continue;
        }
        let record =
            bytes(payload, start, RECORD_SIZE).ok_or(Malformed("FSSM record out of bounds"))?;
        let code = bytes(payload, u64_at(record, 0)?, u64_at(record, 8)?)
            .ok_or(Malformed("native code out of bounds"))?;
        let vgprs = (u32_at(record, RECORD_REGS)? & 0x3FF) as u16;
        if vgprs > 256 {
            return Err(Malformed("implausible VGPR count"));
        }
        let table = |at: usize, count_at: usize| -> Result<&[u8], ShaderError> {
            let len = ENTRY_SIZE * u64::from(u32_at(record, count_at)?);
            if len == 0 {
                return Ok(&[]);
            }
            bytes(payload, u64_at(record, at)?, len)
                .ok_or(Malformed("FSSM root table out of bounds"))
        };
        let layout = table(RECORD_LAYOUT, RECORD_COUNTS)?
            .as_chunks::<{ ENTRY_SIZE as usize }>()
            .0
            .iter()
            .map(|e| RootParam {
                root: RootType::from_raw(le32(e, 0)),
                table_offset_dwords: le32(e, 4),
                sgprs: le32(e, 8),
                base_register: le32(e, 12),
                user_data: e[16],
                slot: e[17],
            })
            .collect();
        let used = table(RECORD_USED, RECORD_COUNTS + 4)?
            .as_chunks::<{ ENTRY_SIZE as usize }>()
            .0
            .iter()
            .map(|e| used_entry(e))
            .collect();
        // The record is in bounds, so its end is too.
        let stage_data = &payload[(start + RECORD_SIZE) as usize..];
        let io = match stage {
            HwStage::Ps => ps_io(stage_data, code.len())?,
            HwStage::Vs => vs_io(stage_data)?,
            HwStage::Cs => StageIo::Cs(ComputeIo {
                rsrc2: u32_at(stage_data, CS_RSRC2)?,
                threads: [
                    u32_at(stage_data, CS_THREADS)?,
                    u32_at(stage_data, CS_THREADS + 4)?,
                    u32_at(stage_data, CS_THREADS + 8)?,
                ],
            }),
            HwStage::Ms => ms_io(stage_data, u32_at(record, RECORD_LDS)?)?,
            _ => StageIo::None,
        };
        out.push(NativeProgram {
            stage,
            wave64: u32_at(record, RECORD_FLAGS)? & FLAG_WAVE32 == 0,
            vgprs,
            layout,
            used,
            io,
            code: code.to_vec(),
        });
    }
    Ok(out)
}

/// The entries of `size` bytes from `at`, with their count at `count_at`.
fn semantics(data: &[u8], at: usize, size: usize, count_at: usize) -> Result<&[u8], ShaderError> {
    let count = u32_at(data, count_at)?;
    if count > MAX_SEMANTICS {
        return Err(Malformed("implausible semantic count"));
    }
    data.get(at..at + size * count as usize)
        .ok_or(Malformed("stage data out of bounds"))
}

fn ps_io(data: &[u8], code_len: usize) -> Result<StageIo, ShaderError> {
    let inputs = semantics(data, PS_INPUTS, 8, PS_INPUT_COUNT)?
        .as_chunks::<8>()
        .0
        .iter()
        .map(|e| PsInput {
            usage: Usage::from_raw(e[0]),
            row: e[1],
            mask: e[2] & 0xf,
            attr: e[3],
            flat: e[7] & 1 != 0,
            fp16: e[7] & 8 != 0,
            pack2: e[7] & 0x10 != 0,
            row2: e[5],
        })
        .collect();
    let size = u32_at(data, PS_PATCH_SIZE)? as usize;
    if size > PATCH_SLOT_DWORDS {
        return Err(Malformed("implausible export patch size"));
    }
    let mut patches = Vec::new();
    for mrt in 0..MRTS {
        let at = PS_PATCHES + mrt * PATCH_BYTES;
        let offset = u32_at(data, at)?;
        if offset == u32::MAX {
            continue;
        }
        if (offset as usize + size) * 4 > code_len {
            return Err(Malformed("export patch outside the code"));
        }
        let full = at + 4 + PATCH_FULL * PATCH_SLOT_DWORDS * 4;
        let code = (0..size)
            .map(|i| u32_at(data, full + 4 * i))
            .collect::<Result<_, _>>()?;
        patches.push(ExportPatch {
            mrt: mrt as u8,
            offset,
            code,
        });
    }
    Ok(StageIo::Ps {
        inputs,
        input_addr: u32_at(data, PS_INPUT_ADDR)?,
        db_shader_control: u32_at(data, PS_DB_SHADER_CONTROL)?,
        patches,
    })
}

fn vs_io(data: &[u8]) -> Result<StageIo, ShaderError> {
    let inputs = semantics(data, VS_INPUTS, 5, VS_INPUT_COUNT)?
        .as_chunks::<5>()
        .0
        .iter()
        .map(|e| VsInput {
            usage: Usage::from_raw(e[0]),
            row: e[1],
            mask: e[2] & 0xf,
            vgpr: e[3],
        })
        .collect();
    let outputs = semantics(data, VS_OUTPUTS, 6, VS_OUTPUT_COUNT)?
        .as_chunks::<6>()
        .0
        .iter()
        .map(|e| VsOutput {
            usage: Usage::from_raw(e[0]),
            row: e[1],
            param: e[2],
            fp16: e[5] & 1 != 0,
            pack2: e[5] & 2 != 0,
        })
        .collect();
    Ok(StageIo::Vs { inputs, outputs })
}

fn ms_io(data: &[u8], lds_bytes: u32) -> Result<StageIo, ShaderError> {
    let StageIo::Vs { outputs, .. } = vs_io(data)? else {
        unreachable!()
    };
    let info = data
        .get(MS_INFO..MS_INFO + 8)
        .ok_or(Malformed("stage data out of bounds"))?;
    let half = |at: usize| u32::from(u16::from_le_bytes([info[at], info[at + 1]]));
    Ok(StageIo::Ms(MeshIo {
        outputs,
        clip_cntl: u32_at(data, MS_CLIP_CNTL)?,
        threads: half(0),
        max_vertices: half(2),
        max_primitives: half(4),
        topology: info[7],
        lds_bytes,
    }))
}

/// One 20-byte descriptor-use entry (FSSM's used table; FSPX's `0x41` records).
fn used_entry(e: &[u8; ENTRY_SIZE as usize]) -> UsedResource {
    UsedResource {
        kind: match e[0] {
            0 => ResourceType::Srv,
            1 => ResourceType::Sampler,
            2 => ResourceType::Cbv,
            3 => ResourceType::Uav,
            n => ResourceType::Other(n),
        },
        root: RootType::from_raw(e[1].into()),
        table_offset_dwords: le32(e, 4),
        lower_bound: le32(e, 8),
        stride: le32(e, 12),
        dwords: e[16],
        user_data: e[17],
        slot: e[18],
    }
}

/// The compute program in an `ACCC` part: `FSPX`, the console driver's own
/// container, which the title's FidelityFX upscaler builds its passes from
/// (no DXIL part). Read from its 29 programs:
///
/// ```text
/// +0x00 'FSPX', version 1, entry count, size; from +0x10 16-byte entries
///       (type, offset, size, 0), offsets from the start of FSPX
/// type 0 (0x1c B): numthreads u16 x3 at +0x0c; +0x10 bit 20 set = wave32
///       (only 32-lane exec code where set, only 64-lane where clear);
///       COMPUTE_PGM_RSRC1 at +0x14, RSRC2 at +0x18
/// type 6: a 0x24-byte prologue (v_nop hash), then the code, padded after
///       the last s_endpgm
/// type 7: 0x18-byte header, then 32-byte records (tag u32, data): tag 0x41
///       holds a used entry as FSSM's (`used_entry`); 0x42 per-view types,
///       0x43 a hash (not read)
/// ```
///
/// User SGPRs s0..s3 are the driver's; the root signature's parameters
/// follow in root order (a table 1 SGPR, a root view 2, constants their
/// count), used or not. `layout` is built from the `RTS0` part that way and
/// checked against every record's SGPR.
pub fn fspx_program(accc: &[u8], rts0: &[u8]) -> Result<NativeProgram, ShaderError> {
    if accc.get(0..4) != Some(b"FSPX") {
        return Err(Malformed("ACCC part lacks FSPX magic"));
    }
    let count = u32_at(accc, 8)? as usize;
    if count > MAX_XDAT_ENTRIES {
        return Err(Malformed("implausible FSPX entry count"));
    }
    let entry = |ty: u32| -> Result<&[u8], ShaderError> {
        for i in 0..count {
            let at = 0x10 + 16 * i;
            if u32_at(accc, at)? == ty {
                let (offset, size) = (u32_at(accc, at + 4)?, u32_at(accc, at + 8)?);
                return bytes(accc, offset.into(), size.into())
                    .ok_or(Malformed("FSPX entry out of bounds"));
            }
        }
        Err(Malformed("FSPX entry missing"))
    };
    let head = entry(0)?;
    let half = |at: usize| crate::dxbc::u16_at(head, at).map(u32::from);
    let threads = [half(0x0c)?, half(0x0e)?, half(0x10)?];
    if threads.contains(&0) || threads.iter().product::<u32>() > 1024 {
        return Err(Malformed("implausible FSPX thread-group size"));
    }
    let wave64 = u32_at(head, 0x10)? & (1 << 20) == 0;
    let rsrc1 = u32_at(head, 0x14)?;
    let rsrc2 = u32_at(head, 0x18)?;
    let code = entry(6)?
        .get(0x24..)
        .ok_or(Malformed("FSPX code shorter than its prologue"))?;
    let end = code
        .as_chunks::<4>()
        .0
        .iter()
        .rposition(|w| u32::from_le_bytes(*w) == S_ENDPGM)
        .ok_or(Malformed("FSPX code without s_endpgm"))?;
    let code = &code[..4 * (end + 1)];
    // RSRC1 VGPRS: granules of 4 registers in wave64, 8 in wave32.
    let vgprs = ((rsrc1 & 0x3f) + 1) * if wave64 { 4 } else { 8 };
    let used: Vec<UsedResource> = entry(7)?
        .get(0x18..)
        .unwrap_or_default()
        .as_chunks::<32>()
        .0
        .iter()
        .filter(|r| le32(*r, 0) == 0x41)
        .map(|r| used_entry(r[4..24].try_into().unwrap()))
        .collect();
    let layout = rts0_layout(rts0, FSPX_DRIVER_SGPRS)?;
    let user_sgprs = rsrc2 >> 1 & 0x1f;
    let end_sgpr = layout.last().map_or(FSPX_DRIVER_SGPRS.into(), |p| u32::from(p.user_data) + p.sgprs);
    if end_sgpr != user_sgprs {
        return Err(Malformed("FSPX root parameters disagree with its user SGPRs"));
    }
    // (static samplers arrive in no SGPR)
    let in_sgprs = used.iter().filter(|u| u.root != RootType::ImmediateSampler);
    if in_sgprs.clone().any(|u| layout.get(u.slot as usize).is_none_or(|p| p.user_data != u.user_data)) {
        return Err(Malformed("FSPX record names another SGPR than its root parameter"));
    }
    Ok(NativeProgram {
        stage: HwStage::Cs,
        wave64,
        vgprs: vgprs as u16,
        layout,
        used,
        io: StageIo::Cs(ComputeIo { rsrc2, threads }),
        code: code.to_vec(),
    })
}

const S_ENDPGM: u32 = 0xBF81_0000;
const FSPX_DRIVER_SGPRS: u8 = 4;

/// One layout entry per root parameter of a serialized root signature
/// (`RTS0`), packed into user SGPRs from `first` in root order.
fn rts0_layout(rts0: &[u8], first: u8) -> Result<Vec<RootParam>, ShaderError> {
    let count = u32_at(rts0, 4)?;
    let params = u32_at(rts0, 8)? as usize;
    if count > 64 {
        return Err(Malformed("implausible root parameter count"));
    }
    let mut next = u32::from(first);
    let mut out = Vec::new();
    for slot in 0..count {
        let at = params + 12 * slot as usize;
        let (root, sgprs) = match u32_at(rts0, at)? {
            0 => (RootType::Table, 1),
            1 => (RootType::Constants, u32_at(rts0, u32_at(rts0, at + 8)? as usize + 8)?),
            2..=4 => (RootType::Descriptor, 2),
            _ => return Err(Malformed("unknown root parameter type")),
        };
        out.push(RootParam {
            root,
            table_offset_dwords: 0,
            sgprs,
            base_register: 0,
            user_data: u8::try_from(next).map_err(|_| Malformed("root parameters past the SGPRs"))?,
            slot: slot as u8,
        });
        next += sgprs;
    }
    Ok(out)
}

/// For entries already sized by `as_chunks`.
fn le32(entry: &[u8], at: usize) -> u32 {
    u32::from_le_bytes(entry[at..at + 4].try_into().unwrap())
}

/// `data[offset..offset + len]` with untrusted 64-bit bounds.
fn bytes(data: &[u8], offset: u64, len: u64) -> Option<&[u8]> {
    let start = usize::try_from(offset).ok()?;
    data.get(start..start.checked_add(usize::try_from(len).ok()?)?)
}

#[cfg(test)]
mod tests {
    use super::*;

    const VS_CODE: [u8; 8] = [1, 2, 3, 4, 5, 6, 7, 8];
    const PS_CODE: [u8; 4] = [9, 10, 11, 12];
    const VGPRS: u32 = 28;

    fn slot(stage: HwStage) -> usize {
        SLOTS.iter().find(|s| s.0 == stage).unwrap().1
    }

    /// Stage data: no semantics, and every export patch absent.
    const STAGE_DATA: usize = PS_PATCHES + MRTS * PATCH_BYTES;

    fn empty_stage_data() -> Vec<u8> {
        let mut data = vec![0; STAGE_DATA];
        for mrt in 0..MRTS {
            let at = PS_PATCHES + mrt * PATCH_BYTES;
            data[at..at + 4].copy_from_slice(&u32::MAX.to_le_bytes());
        }
        data
    }

    /// Where a one-program blob's stage data starts.
    fn stage_data_at(blob: &[u8], stage: HwStage) -> usize {
        FSSM_HEADER
            + u64_at(blob, FSSM_HEADER + slot(stage)).unwrap() as usize
            + RECORD_SIZE as usize
    }

    /// FSSM blob with `programs` as (stage, code) pairs.
    fn fssm(programs: &[(HwStage, &[u8])]) -> Vec<u8> {
        let mut payload = GRAPHICS_SLOT_COUNT.to_le_bytes().to_vec();
        payload.resize(slot(HwStage::Cs) + 8, 0);
        for (_, at) in SLOTS {
            payload[at..at + 8].copy_from_slice(&ABSENT.to_le_bytes());
        }
        for &(stage, code) in programs {
            let slot = slot(stage);
            let record = payload.len() as u64;
            payload[slot..slot + 8].copy_from_slice(&record.to_le_bytes());
            let code_at = record + RECORD_SIZE + STAGE_DATA as u64;
            payload.extend_from_slice(&code_at.to_le_bytes());
            payload.extend_from_slice(&(code.len() as u64).to_le_bytes());
            let flags = if stage == HwStage::Ps { FLAG_WAVE32 } else { 0 };
            payload.extend_from_slice(&[0; 4]);
            payload.extend_from_slice(&flags.to_le_bytes());
            payload.resize(record as usize + RECORD_REGS, 0);
            payload.extend_from_slice(&(VGPRS | 31 << 20).to_le_bytes());
            payload.extend_from_slice(&empty_stage_data());
            payload.extend_from_slice(code);
        }
        let mut out = b"FSSM".to_vec();
        out.extend_from_slice(&FSSM_VERSION.to_le_bytes());
        out.extend_from_slice(&(payload.len() as u64).to_le_bytes());
        out.extend_from_slice(&payload);
        out
    }

    fn xbch(unpacked: &[u8]) -> Vec<u8> {
        let mut buf = vec![0u8; zlib_rs::compress_bound(unpacked.len())];
        let (stream, rc) =
            zlib_rs::compress_slice(&mut buf, unpacked, zlib_rs::DeflateConfig::new(6));
        assert_eq!(rc, zlib_rs::ReturnCode::Ok);
        let mut out = b"XBCH".to_vec();
        out.extend_from_slice(&XBCH_VERSION.to_le_bytes());
        out.extend_from_slice(&(stream.len() as u64).to_le_bytes());
        out.extend_from_slice(&(unpacked.len() as u64).to_le_bytes());
        out.extend_from_slice(stream);
        out
    }

    /// `DXIL` part data: program header, XDAT index, then the blobs.
    fn part(blobs: &[(u32, &[u8])]) -> Vec<u8> {
        let mut out = vec![0u8; XDAT_OFFSET];
        out.extend_from_slice(b"XDAT");
        out.extend_from_slice(&(blobs.len() as u32).to_le_bytes());
        out.extend_from_slice(&0x24u32.to_le_bytes());
        let mut offset = out.len() + 12 * blobs.len();
        for (kind, blob) in blobs {
            for v in [*kind, offset as u32, blob.len() as u32] {
                out.extend_from_slice(&v.to_le_bytes());
            }
            offset += blob.len();
        }
        for (_, blob) in blobs {
            out.extend_from_slice(blob);
        }
        out
    }

    #[test]
    fn finds_graphics_programs() {
        let native = xbch(&fssm(&[(HwStage::Vs, &VS_CODE), (HwStage::Ps, &PS_CODE)]));
        let data = part(&[(0x0a, b"xbop"), (KIND_NATIVE, &native)]);
        let found = native_programs(&data).unwrap();
        assert_eq!(
            found,
            [
                NativeProgram {
                    stage: HwStage::Vs,
                    wave64: true,
                    vgprs: 28,
                    layout: vec![],
                    used: vec![],
                    io: StageIo::Vs {
                        inputs: vec![],
                        outputs: vec![],
                    },
                    code: VS_CODE.to_vec()
                },
                NativeProgram {
                    stage: HwStage::Ps,
                    wave64: false,
                    vgprs: 28,
                    layout: vec![],
                    used: vec![],
                    io: StageIo::Ps {
                        inputs: vec![],
                        input_addr: 0,
                        db_shader_control: 0,
                        patches: vec![],
                    },
                    code: PS_CODE.to_vec()
                },
            ]
        );
    }

    #[test]
    fn reads_root_tables() {
        let mut blob = fssm(&[(HwStage::Ps, &PS_CODE)]);
        let record = FSSM_HEADER + u64_at(&blob, FSSM_HEADER + slot(HwStage::Ps)).unwrap() as usize;
        let tables = (blob.len() - FSSM_HEADER) as u64;
        // A table range in s9 (root slot 1), and an SRV read from it.
        for v in [3u32, 2, 1, 5] {
            blob.extend_from_slice(&v.to_le_bytes());
        }
        blob.extend_from_slice(&[9, 1, 0, 0, 0, 3, 0, 0]);
        for v in [2u32, 5, 4] {
            blob.extend_from_slice(&v.to_le_bytes());
        }
        blob.extend_from_slice(&[8, 9, 1, 0]);
        let payload = (blob.len() - FSSM_HEADER) as u64;
        blob[8..16].copy_from_slice(&payload.to_le_bytes());
        blob[record + RECORD_LAYOUT..][..8].copy_from_slice(&tables.to_le_bytes());
        blob[record + RECORD_USED..][..8].copy_from_slice(&(tables + ENTRY_SIZE).to_le_bytes());
        blob[record + RECORD_COUNTS..][..8].copy_from_slice(&[1, 0, 0, 0, 1, 0, 0, 0]);

        let p = &native_programs(&part(&[(KIND_NATIVE, &xbch(&blob))])).unwrap()[0];
        assert_eq!(
            p.layout,
            [RootParam {
                root: RootType::Table,
                table_offset_dwords: 2,
                sgprs: 1,
                base_register: 5,
                user_data: 9,
                slot: 1
            }]
        );
        assert_eq!(
            p.used,
            [UsedResource {
                kind: ResourceType::Srv,
                root: RootType::Table,
                table_offset_dwords: 2,
                lower_bound: 5,
                stride: 4,
                dwords: 8,
                user_data: 9,
                slot: 1
            }]
        );

        blob[record + RECORD_COUNTS] = 9;
        assert!(native_programs(&part(&[(KIND_NATIVE, &xbch(&blob))])).is_err());
    }

    #[test]
    fn reads_pixel_io_and_applies_the_full_export_patch() {
        let code = [0xBF80_0000u32; 4]; // s_nop
        let bytes: Vec<u8> = code.iter().flat_map(|w| w.to_le_bytes()).collect();
        let mut blob = fssm(&[(HwStage::Ps, &bytes)]);
        let e = stage_data_at(&blob, HwStage::Ps);
        // TEXCOORD in row 1 (xy), read as attr0, flat.
        blob[e + PS_INPUTS..][..8].copy_from_slice(&[6, 1, 3, 0, 0, 0, 1, 1]);
        blob[e + PS_INPUT_COUNT..][..4].copy_from_slice(&1u32.to_le_bytes());
        blob[e + PS_INPUT_ADDR..][..4].copy_from_slice(&2u32.to_le_bytes());
        blob[e + PS_PATCH_SIZE..][..4].copy_from_slice(&2u32.to_le_bytes());
        // MRT1 at dword 1: the 32_ABGR variant is `exp mrt1 v0-v3 done`.
        let patch = e + PS_PATCHES + PATCH_BYTES;
        blob[patch..][..4].copy_from_slice(&1u32.to_le_bytes());
        let full = patch + 4 + PATCH_FULL * PATCH_SLOT_DWORDS * 4;
        blob[full..][..8].copy_from_slice(&[0x1f, 0x18, 0, 0xf8, 0, 1, 2, 3]);

        let p = &native_programs(&part(&[(KIND_NATIVE, &xbch(&blob))])).unwrap()[0];
        let StageIo::Ps {
            inputs,
            input_addr,
            patches,
            ..
        } = &p.io
        else {
            panic!("{:?}", p.io)
        };
        let input = PsInput {
            usage: Usage::Generic,
            row: 1,
            mask: 3,
            attr: 0,
            flat: true,
            fp16: false,
            pack2: false,
            row2: 0,
        };
        assert_eq!((&inputs[..], *input_addr), (&[input][..], 2));
        assert_eq!(patches.len(), 1);
        assert_eq!((patches[0].mrt, patches[0].offset), (1, 1));
        let patched = p.patched_code();
        assert_eq!(patched[..4], p.code[..4]);
        assert_eq!(patched[4..12], [0x1f, 0x18, 0, 0xf8, 0, 1, 2, 3]);
        assert_eq!(patched[12..], p.code[12..]);

        blob[patch..][..4].copy_from_slice(&3u32.to_le_bytes());
        assert!(native_programs(&part(&[(KIND_NATIVE, &xbch(&blob))])).is_err());
    }

    #[test]
    fn finds_mesh_and_compute_programs() {
        for stage in [HwStage::Ms, HwStage::Cs] {
            let data = part(&[(KIND_NATIVE, &xbch(&fssm(&[(stage, &VS_CODE)])))]);
            let found = native_programs(&data).unwrap();
            assert_eq!(found.len(), 1);
            assert_eq!(found[0].stage, stage);
        }
    }

    #[test]
    fn reads_mesh_outputs_and_limits() {
        let mut blob = fssm(&[(HwStage::Ms, &VS_CODE)]);
        let e = stage_data_at(&blob, HwStage::Ms);
        // param0 is row 1 per vertex, param2 row 0 of the primitive signature.
        blob[e + VS_OUTPUTS..][..12].copy_from_slice(&[6, 1, 0, 0, 0, 0, 6, 0, 0x82, 0, 0, 0]);
        blob[e + VS_OUTPUT_COUNT..][..4].copy_from_slice(&2u32.to_le_bytes());
        blob[e + MS_CLIP_CNTL..][..4].copy_from_slice(&0x0040_000Fu32.to_le_bytes());
        // 128 lanes, 128 vertices, 64 primitives, 13 user SGPRs, triangles.
        blob[e + MS_INFO..][..8].copy_from_slice(&[0x80, 0, 0x80, 0, 0x40, 0, 13, 2]);
        let record = e - RECORD_SIZE as usize;
        blob[record + RECORD_LDS..][..4].copy_from_slice(&16u32.to_le_bytes());
        let p = &native_programs(&part(&[(KIND_NATIVE, &xbch(&blob))])).unwrap()[0];
        let out = |row, param| VsOutput {
            usage: Usage::Generic,
            row,
            param,
            fp16: false,
            pack2: false,
        };
        assert_eq!(
            p.io,
            StageIo::Ms(MeshIo {
                outputs: vec![out(1, 0), out(0, 0x82)],
                clip_cntl: 0x0040_000F,
                threads: 128,
                max_vertices: 128,
                max_primitives: 64,
                topology: 2,
                lds_bytes: 16,
            })
        );
    }

    #[test]
    fn compute_dispatch_settings() {
        // 5 user SGPRs, group id x and y, thread ids in v0-v1, 1 KiB LDS.
        let cs = ComputeIo {
            rsrc2: 0x1098A,
            threads: [16, 16, 1],
        };
        assert_eq!(cs.user_sgprs(), 5);
        assert_eq!(cs.system_sgprs(), ([Some(5), Some(6), None], None));
        assert_eq!((cs.thread_id_vgprs(), cs.lds_bytes()), (2, 1024));
        // 6 user SGPRs, all three group ids, v0-v2, no LDS.
        let cs = ComputeIo {
            rsrc2: 0x138C,
            threads: [8, 8, 1],
        };
        assert_eq!(cs.system_sgprs(), ([Some(6), Some(7), Some(8)], None));
        assert_eq!((cs.thread_id_vgprs(), cs.lds_bytes()), (3, 0));
    }

    #[test]
    fn initialized_sgprs_covers_user_system_and_scratch() {
        // 5 user SGPRs, group ids x and y (s5, s6), no scratch: s0..=s6.
        let cs = ComputeIo {
            rsrc2: 0x1098A,
            threads: [16, 16, 1],
        };
        assert_eq!(cs.initialized_sgprs(), 7);
        // All three group ids after 6 user SGPRs, plus the info SGPR: s0..=s9.
        let cs = ComputeIo {
            rsrc2: 0x78C,
            threads: [8, 8, 1],
        };
        assert_eq!(cs.initialized_sgprs(), 10);
        // The same with scratch enabled adds the wave offset: s0..=s10.
        let cs = ComputeIo {
            rsrc2: 0x78D,
            threads: [8, 8, 1],
        };
        assert_eq!(cs.initialized_sgprs(), 11);
    }

    #[test]
    fn reads_every_native_blob() {
        let es = xbch(&fssm(&[(HwStage::Es, &VS_CODE)]));
        let vs = xbch(&fssm(&[(HwStage::Vs, &VS_CODE)]));
        let data = part(&[(0x12, &[0; 16]), (KIND_NATIVE, &es), (KIND_NATIVE, &vs)]);
        let stages: Vec<_> = native_programs(&data)
            .unwrap()
            .iter()
            .map(|p| p.stage)
            .collect();
        assert_eq!(stages, [HwStage::Es, HwStage::Vs]);
    }

    #[test]
    fn pc_layout_has_none() {
        assert!(native_programs(&[0u8; 64]).unwrap().is_empty());
        assert!(
            native_programs(&part(&[(0x0a, b"xbop")]))
                .unwrap()
                .is_empty()
        );
    }

    #[test]
    fn rejects_corrupt_blobs() {
        let good = xbch(&fssm(&[(HwStage::Ls, &VS_CODE)]));
        let corrupt = |at: usize, byte: u8| {
            let mut blob = good.clone();
            blob[at] = byte;
            native_programs(&part(&[(KIND_NATIVE, &blob)]))
        };
        assert!(corrupt(4, 9).is_err(), "unknown version");
        assert!(corrupt(16, 0xFF).is_err(), "stated size");
        assert!(corrupt(XBCH_HEADER + 2, 0xFF).is_err(), "zlib stream");
        assert!(native_programs(&part(&[(KIND_NATIVE, &good[..20])])).is_err());

        let mut bad_record = fssm(&[(HwStage::Ls, &VS_CODE)]);
        let at = FSSM_HEADER + slot(HwStage::Ls);
        bad_record[at..at + 8].copy_from_slice(&u64::MAX.to_le_bytes());
        assert!(native_programs(&part(&[(KIND_NATIVE, &xbch(&bad_record))])).is_err());
    }

    #[test]
    fn rejects_truncated_oversized_and_bad_checksum_xbch_streams() {
        let plain = fssm(&[(HwStage::Vs, &VS_CODE)]);
        let good = xbch(&plain);
        assert_eq!(unpack(&good).unwrap(), plain);

        // Truncated zlib stream (drop last byte, update packed length).
        let mut truncated = good[..good.len() - 1].to_vec();
        let new_packed = (truncated.len() - XBCH_HEADER) as u64;
        truncated[8..16].copy_from_slice(&new_packed.to_le_bytes());
        assert!(unpack(&truncated).is_err());

        // Stated unpacked size smaller than the inflated output.
        let mut oversized = good.clone();
        oversized[16..24].copy_from_slice(&((plain.len() - 1) as u64).to_le_bytes());
        assert!(unpack(&oversized).is_err());

        // Stated unpacked size larger than the inflated output.
        let mut underfilled = good.clone();
        underfilled[16..24].copy_from_slice(&((plain.len() + 1) as u64).to_le_bytes());
        assert!(unpack(&underfilled).is_err());

        // Corrupt Adler-32 trailer.
        let mut bad_adler = good.clone();
        *bad_adler.last_mut().unwrap() ^= 1;
        assert!(unpack(&bad_adler).is_err());
    }

    /// An FSPX part as FidelityFX's: wave32, 32 threads, a UAV table, an
    /// SRV table and a root CBV (s4, s5, s6..s7).
    fn fspx(flags: u32, rsrc2: u32, record_sgpr: u8) -> (Vec<u8>, Vec<u8>) {
        let mut head = vec![0u8; 0x1c];
        head[0x0c..0x12].copy_from_slice(&[32, 0, 1, 0, 1, 0]);
        head[0x10..0x14].copy_from_slice(&(flags | 1).to_le_bytes());
        head[0x14..0x18].copy_from_slice(&0x402c_0341u32.to_le_bytes());
        head[0x18..0x1c].copy_from_slice(&rsrc2.to_le_bytes());
        let mut code = vec![0u8; 0x24];
        code.extend(0xBF80_0000u32.to_le_bytes()); // s_nop
        code.extend(S_ENDPGM.to_le_bytes());
        code.extend(0xBF9F_0000u32.to_le_bytes()); // s_code_end padding
        let mut decl = vec![0u8; 0x18];
        let mut rec = vec![0u8; 32];
        rec[0] = 0x41;
        rec[4..8].copy_from_slice(&[3, 3, 0, 0]); // UAV in a table
        rec[12] = 11;
        rec[20..23].copy_from_slice(&[8, record_sgpr, 0]);
        decl.extend(&rec);
        let entries = [(0u32, head), (6, code), (7, decl)];
        let mut part = Vec::from(*b"FSPX");
        part.extend(1u32.to_le_bytes());
        part.extend((entries.len() as u32).to_le_bytes());
        part.extend(0u32.to_le_bytes());
        let mut body: Vec<u8> = Vec::new();
        let base = 0x10 + 16 * entries.len();
        for (ty, data) in &entries {
            part.extend(ty.to_le_bytes());
            part.extend(((base + body.len()) as u32).to_le_bytes());
            part.extend((data.len() as u32).to_le_bytes());
            part.extend(0u32.to_le_bytes());
            body.extend(data);
        }
        part.extend(body);
        // RTS0 v2: table, table, CBV.
        let mut rts0 = Vec::new();
        for v in [2u32, 3, 0x18, 0, 0, 0] {
            rts0.extend(v.to_le_bytes());
        }
        for (ty, payload) in [(0u32, 0x3cu32), (0, 0x3c), (2, 0x3c)] {
            rts0.extend([ty, 0, payload].iter().flat_map(|v| v.to_le_bytes()));
        }
        rts0.extend([0u32; 4].iter().flat_map(|v| v.to_le_bytes()));
        (part, rts0)
    }

    #[test]
    fn fspx_program_reads_driver_compute_container() {
        let (part, rts0) = fspx(1 << 20, 8 << 1, 4);
        let p = fspx_program(&part, &rts0).unwrap();
        assert_eq!(p.stage, HwStage::Cs);
        assert!(!p.wave64);
        assert_eq!(p.vgprs, 16);
        assert_eq!(p.code.len(), 8, "prologue and padding cut off");
        assert_eq!(
            p.layout.iter().map(|r| (r.root, r.user_data, r.sgprs, r.slot)).collect::<Vec<_>>(),
            [(RootType::Table, 4, 1, 0), (RootType::Table, 5, 1, 1), (RootType::Descriptor, 6, 2, 2)]
        );
        assert_eq!(p.used.len(), 1);
        assert_eq!((p.used[0].kind, p.used[0].lower_bound, p.used[0].dwords), (ResourceType::Uav, 11, 8));
        let StageIo::Cs(io) = p.io else { panic!() };
        assert_eq!(io.threads, [32, 1, 1]);
        // Wave64 without the flag; refused when SGPRs or records disagree.
        assert!(fspx_program(&fspx(0, 8 << 1, 4).0, &rts0).unwrap().wave64);
        assert!(fspx_program(&fspx(0, 9 << 1, 4).0, &rts0).is_err());
        assert!(fspx_program(&fspx(0, 8 << 1, 5).0, &rts0).is_err());
    }

    /// `VICTORY_SAMPLE_FSPX=<file>`: translates every FSPX compute container
    /// in that file (e.g. the title's FidelityFX DLL) and prints the result.
    #[test]
    fn fspx_sample_translates() {
        let Some(path) = std::env::var_os("VICTORY_SAMPLE_FSPX") else {
            return;
        };
        let data = std::fs::read(path).unwrap();
        #[cfg(windows)]
        let host = std::env::var_os("VICTORY_SAMPLE_FSPX_HOST")
            .and_then(|_| crate::emit::host::Host::new(false).ok());
        let (mut ok, mut n, mut at) = (0, 0, 0);
        while let Some(i) = data[at..].windows(4).position(|w| w == b"DXBC").map(|i| at + i) {
            at = i + 4;
            let Ok(size) = u32_at(&data, i + 24) else { break };
            let Some(blob) = data.get(i..i + size as usize) else { continue };
            let Ok(c) = crate::Container::parse(blob) else { continue };
            if c.part(crate::FourCC::ACCC).is_none() {
                continue;
            }
            n += 1;
            // The hash the layer logs for a shader (blake3, first 16 hex digits).
            print!("[{}] ", &blake3::hash(blob).to_hex()[..16]);
            match crate::emit::translate(blob) {
                Ok(s) => {
                    ok += 1;
                    // On this PC's card too (VICTORY_SAMPLE_FSPX_HOST=1).
                    #[cfg(windows)]
                    let made = match &host {
                        Some(h) => match h.compute_pipeline(&s) {
                            Ok(_) => ", pipeline made".to_owned(),
                            Err(e) => format!(", PIPELINE REFUSED {e}"),
                        },
                        None => String::new(),
                    };
                    #[cfg(not(windows))]
                    let made = "";
                    println!("FSPX @{i:#x}: {} B DXIL{made}", s.dxil.len());
                }
                Err(e) => println!("FSPX @{i:#x}: FAILED {e}"),
            }
            // VICTORY_SAMPLE_FSPX_DIS=<container offset>: its code (translated or not), resources and layout.
            if std::env::var("VICTORY_SAMPLE_FSPX_DIS").ok() == Some(format!("{i:#x}")) {
                let accc = c.part(crate::FourCC::ACCC).unwrap().data;
                let rts0 = c.part(crate::FourCC::RTS0).unwrap().data;
                let p = fspx_program(accc, rts0).unwrap();
                println!("  used {:?}
  layout {:?}", p.used, p.layout);
                // VICTORY_SAMPLE_FSPX_OUT=<file>: its translated DXIL.
                if let (Some(out), Ok(s)) = (std::env::var_os("VICTORY_SAMPLE_FSPX_OUT"), crate::emit::translate(blob)) {
                    std::fs::write(out, &s.dxil).unwrap();
                }
                for inst in crate::gfx10::decode(&p.code).unwrap() {
                    println!("  {:#06x} {}", inst.offset, inst.text(p.wave64));
                }
            }
        }
        println!("FSPX: {ok} / {n} translated");
        assert!(n > 0, "no FSPX containers in the file");
    }

    /// `VICTORY_SAMPLE_FSPX=<FidelityFX DLL>`: its luminance pass (ba578b59, the SPD pass whose
    /// last workgroup writes the exposure) run on this PC's card over a grey 1280x720 input: the
    /// counter must come back to 0 and the exposure buffer hold a finite, positive value (run
    /// victory26: it stayed [0, 0] and the upscaled picture went white).
    #[cfg(windows)]
    #[test]
    fn fspx_luminance_pass_writes_exposure() {
        use crate::emit::host::{Dispatch, Host, Root, View};
        let Some(path) = std::env::var_os("VICTORY_SAMPLE_FSPX") else {
            return;
        };
        let data = std::fs::read(path).unwrap();
        let at = 0x704f0;
        let size = u32_at(&data, at + 24).unwrap() as usize;
        let blob = &data[at..at + size];
        assert_eq!(&blake3::hash(blob).to_hex()[..16], "ba578b59c8c28a57");
        let shader = crate::emit::translate(blob).unwrap();
        let Ok(host) = Host::new(false) else {
            eprintln!("no SM 6.6 device; skipped");
            return;
        };
        let (root, pso) = host.compute_pipeline(&shader).unwrap();
        let grey = vec![[0.18f32, 0.18, 0.18, 1.0]; 1280 * 720];
        // Heap: table 0 = counter, luminance level, exposure; table 1 = the colour input.
        let views = [View::UintTextureUav(1, 1), View::TextureUav(80, 45), View::RawUav(2), View::Texture(1280, 720, &grey)];
        let mut ud = vec![0u32; shader.user_data_dwords().max(1) as usize];
        for u in &shader.user_data {
            ud[u.index as usize] = match (u.root, u.slot) {
                (RootType::Table, 1) => 3 * 32,
                // the exposure view's title address, as run victory28 bound it
                (RootType::Other(0x101), 0) => 0x5a4c_0000,
                (RootType::Other(0x102), 0) => 0x291,
                _ => 0,
            };
        }
        for mips in [10u32, 11] {
            // mips, work groups (20 x 12 tiles of 64x64), group offset
            let mut cb = vec![0u32; 64];
            cb[..4].copy_from_slice(&[mips, 240, 0, 0]);
            cb[6] = 1f32.to_bits(); // pre-exposure (the pass divides by it)
            let out = host
                .dispatch(
                    &root,
                    &pso,
                    &Dispatch {
                        roots: vec![(0, Root::Constants(ud.clone())), (1, Root::View { srv: false, data: cb })],
                        views: &views,
                        samplers: &[],
                        groups: [20, 12, 1],
                    },
                )
                .unwrap();
            let exposure = f32::from_bits(out[2][0]);
            println!("mips {mips}: counter {:?} exposure {exposure} (buffer {:x?})", out[0], out[2]);
            assert_eq!(out[0][0], 0, "mips {mips}: the last group resets the counter");
            assert!(exposure.is_finite() && exposure > 0.0, "mips {mips}: exposure {exposure}");
        }
    }

    /// `VICTORY_SAMPLE_FSPX=<FidelityFX DLL>`: its upscale pass (fdce2a0a, the one that turned
    /// run victory28's picture white at frame 2400 from a sane 1280x720 input and history) run
    /// on this PC's card with that frame's constants over grey inputs: the 1920x1080 output
    /// must stay finite and near the input's grey.
    #[cfg(windows)]
    #[test]
    fn fspx_upscale_pass_keeps_grey() {
        use crate::emit::host::{Dispatch, Host, Root, Stride, View};
        let Some(path) = std::env::var_os("VICTORY_SAMPLE_FSPX") else {
            return;
        };
        let data = std::fs::read(path).unwrap();
        let at = 0x9fe90;
        let size = u32_at(&data, at + 24).unwrap() as usize;
        let blob = &data[at..at + size];
        assert_eq!(&blake3::hash(blob).to_hex()[..16], "fdce2a0a9fb1521a");
        let shader = crate::emit::translate(blob).unwrap();
        let Ok(host) = Host::new(std::env::var_os("VICTORY_PROBE_WARP").is_some()) else {
            eprintln!("no SM 6.6 device; skipped");
            return;
        };
        let (root, pso) = host.compute_pipeline(&shader).unwrap();
        let grey = 0.18f32;
        let input = vec![[grey, grey, grey, 1.0]; 1280 * 720];
        let history = vec![[grey, grey, grey, 1.0]; 1920 * 1080];
        let exposure = [std::env::var("VICTORY_PROBE_EXPOSURE").ok().and_then(|v| v.parse::<f32>().ok()).unwrap_or(1.0).to_bits(), 0];
        let pad = [0u32; 4];
        // Table 0 (base 0): UAVs at 1, 2, 6, 11; table 1 (base 12): SRVs at 15, 21, 29.
        let mut views: Vec<View> = (0..30).map(|_| View::Cbv(&pad)).collect();
        views[1] = View::TextureUav(1920, 1080);
        views[2] = View::TextureUav(1920, 1080);
        views[6] = View::TextureUav(1920, 1080);
        // VICTORY_PROBE_SCRATCH=<seed>: the network's scratch filled with pseudo-random bytes.
        let scratch: Vec<u32> = match std::env::var("VICTORY_PROBE_SCRATCH").ok().and_then(|v| v.parse::<u32>().ok()) {
            Some(mut x) => (0..21_045_760 / 4).map(|_| { x ^= x << 13; x ^= x >> 17; x ^= x << 5; x }).collect(),
            // VICTORY_PROBE_SCRATCH_FILE=<frame-N-buffer-...-21045760.bin>: the game's own, as dumped.
            None => match std::env::var_os("VICTORY_PROBE_SCRATCH_FILE") {
                Some(f) => std::fs::read(f).unwrap().chunks_exact(4).map(|c| u32::from_le_bytes(c.try_into().unwrap())).collect(),
                None => vec![0; 21_045_760 / 4],
            },
        };
        views[11] = View::RawUavData(&scratch);
        views[15] = View::Texture(1280, 720, &input);
        views[21] = View::Texture(1920, 1080, &history);
        views[29] = View::Buffer { data: &exposure, stride: Stride::Raw };
        let mut ud = vec![0u32; shader.user_data_dwords().max(1) as usize];
        for u in &shader.user_data {
            ud[u.index as usize] = match (u.root, u.slot) {
                (RootType::Table, 1) => 12 * 32,
                // the big buffer's title address, as run victory30 bound it
                (RootType::Other(0x101), _) => 0x7b79_0000,
                (RootType::Other(0x102), _) => 0x185,
                _ => 0,
            };
        }
        let mut b0 = vec![0u32; 64];
        b0[..UPSCALE_B0.len()].copy_from_slice(&UPSCALE_B0);
        let mut b1 = vec![0u32; 64];
        b1[..UPSCALE_B1.len()].copy_from_slice(&UPSCALE_B1);
        let out = host
            .dispatch(
                &root,
                &pso,
                &Dispatch {
                    roots: vec![
                        (0, Root::Constants(ud)),
                        (1, Root::View { srv: false, data: b0 }),
                        (2, Root::View { srv: false, data: b1 }),
                    ],
                    views: &views,
                    samplers: &[],
                    groups: [60, 34, 1],
                },
            )
            .unwrap();
        // out: the UAVs in heap order: 1, 2, 6, 11.
        for (k, name) in [(0, "u1"), (1, "u2 (the white one)"), (2, "u6")] {
            let px: Vec<f32> = out[k].iter().map(|&b| f32::from_bits(b)).collect();
            let bad = px.iter().filter(|v| !v.is_finite()).count();
            let mid = &px[(540 * 1920 + 960) * 4..][..4];
            let white = px.iter().filter(|&&v| v >= 64000.0).count();
            println!("{name}: {bad} non-finite, {white} at 64000, of {}; centre {mid:?}", px.len());
        }
        let px: Vec<f32> = out[1].iter().map(|&b| f32::from_bits(b)).collect();
        let mid = &px[(540 * 1920 + 960) * 4..][..3];
        assert!(mid.iter().all(|v| v.is_finite() && (v - grey).abs() < 0.1), "centre {mid:?}");
    }

    /// `VICTORY_REPLAY_SPEC=<spec>` (made from a frame's ops listing), `VICTORY_SAMPLE_FSPX=<DLL>`,
    /// `VICTORY_REPLAY_SIDE=<the 128 KiB side buffer>`: FidelityFX's passes of one frame run in
    /// order on this PC's card over grey inputs, sharing the network's 21 MB scratch; after each,
    /// how many scratch bytes it changed and how many of those are saturated (0x7f / 0x80).
    #[cfg(windows)]
    #[test]
    fn fspx_replay_frame() {
        use crate::emit::host::{Dispatch, Host, Root, Stride, View};
        let (Some(spec), Some(dll)) = (std::env::var_os("VICTORY_REPLAY_SPEC"), std::env::var_os("VICTORY_SAMPLE_FSPX")) else {
            return;
        };
        let data = std::fs::read(dll).unwrap();
        let side: Vec<u32> = std::env::var_os("VICTORY_REPLAY_SIDE")
            .map(|f| std::fs::read(f).unwrap().chunks_exact(4).map(|c| u32::from_le_bytes(c.try_into().unwrap())).collect())
            .unwrap_or_else(|| vec![0; 32768]);
        let host = Host::new(false).expect("no SM 6.6 device");
        let mut scratch = vec![0u32; 21_045_760 / 4];
        let spec = std::fs::read_to_string(spec).unwrap();
        let pad = [0u32; 4];
        let exposure = [5.194174f32.to_bits(), 5.1845107f32.to_bits()];
        for pass in spec.split("\nend").filter(|p| p.contains("pass ")) {
            let mut lines = pass.lines().map(str::trim).filter(|l| !l.is_empty());
            let head: Vec<&str> = lines.next().unwrap().split_whitespace().collect();
            let (hash, at) = (head[1], usize::from_str_radix(&head[2][2..], 16).unwrap());
            let groups = [head[3].parse().unwrap(), head[4].parse().unwrap(), head[5].parse().unwrap()];
            let size = u32_at(&data, at + 24).unwrap() as usize;
            let blob = &data[at..at + size];
            assert_eq!(&blake3::hash(blob).to_hex()[..16], hash);
            let shader = crate::emit::translate(blob).unwrap();
            if let Some(dir) = std::env::var_os("VICTORY_REPLAY_DXIL") {
                std::fs::write(std::path::Path::new(&dir).join(format!("{hash}.dxil")), &shader.dxil).unwrap();
            }
            let (root, pso) = host.compute_pipeline(&shader).unwrap();
            let (mut uds, mut cbs, mut views_spec) = (Vec::new(), Vec::new(), Vec::new());
            for l in lines {
                let w: Vec<&str> = l.split_whitespace().collect();
                match w[0] {
                    "ud" => uds.push((w[1].parse::<u32>().unwrap(), u32::from_str_radix(&w[2][2..], 16).unwrap())),
                    "cb" => cbs.push((w[1].parse::<u8>().unwrap(), w[2..].iter().map(|x| u32::from_str_radix(x, 16).unwrap()).collect::<Vec<_>>())),
                    "view" => views_spec.push(w[1..].iter().map(|x| x.to_string()).collect::<Vec<_>>()),
                    _ => {}
                }
            }
            // Table slot k starts at heap index 64 k; a view at dword d of it is index 64 k + d / 8.
            let index = |v: &[String]| 64 * v[1].parse::<usize>().unwrap() + v[2].parse::<usize>().unwrap() / 8;
            let n = views_spec.iter().map(|v| index(v) + 1).max().unwrap_or(1);
            let texels: Vec<Vec<[f32; 4]>> = views_spec
                .iter()
                .map(|v| match (v[0].as_str(), v[3].as_str()) {
                    ("Srv", "texture") => {
                        let (w, h, f) = (v[4].parse::<usize>().unwrap(), v[5].parse::<usize>().unwrap(), v[6].as_str());
                        // colour grey, motion (34: R16G16) still, depth (19) halfway
                        let t = match f { "34" => [0.0; 4], "19" => [0.5; 4], _ => [0.18, 0.18, 0.18, 1.0] };
                        vec![t; w * h]
                    }
                    _ => Vec::new(),
                })
                .collect();
            let mut views: Vec<View> = (0..n).map(|_| View::Cbv(&pad)).collect();
            let mut scratch_at = None;
            for (v, t) in views_spec.iter().zip(&texels) {
                let i = index(v);
                let num = |k: usize| v[k].parse::<u32>().unwrap();
                views[i] = match (v[0].as_str(), v[3].as_str()) {
                    ("Uav", "texture") if v[6] == "42" => View::UintTextureUav(num(4), num(5)),
                    ("Uav", "texture") => View::TextureUav(num(4), num(5)),
                    ("Uav", "buffer") if num(5) == 21_045_760 => {
                        scratch_at = Some(i);
                        View::RawUavData(&scratch)
                    }
                    ("Uav", "buffer") => View::RawUav(num(5) / 4),
                    ("Srv", "texture") => View::Texture(num(4), num(5), t),
                    ("Srv", "buffer") if num(5) == 8 => View::Buffer { data: &exposure, stride: Stride::Raw },
                    ("Srv", "buffer") if num(5) == 131_072 => View::Buffer { data: &side, stride: Stride::Raw },
                    other => panic!("{hash}: a view the replay does not make: {other:?}"),
                };
            }
            let mut ud = vec![0u32; shader.user_data_dwords().max(1) as usize];
            for u in &shader.user_data {
                ud[u.index as usize] = match u.root {
                    RootType::Table => 64 * 32 * u32::from(u.slot),
                    _ => uds.iter().find(|x| x.0 == u.index).map_or(0, |x| x.1),
                };
            }
            let mut slots = shader.root_cbvs.clone();
            slots.sort_unstable();
            let mut roots = vec![(0, Root::Constants(ud))];
            for (k, slot) in slots.iter().enumerate() {
                let mut d = cbs.iter().find(|c| c.0 == *slot).map(|c| c.1.clone()).unwrap_or_else(|| {
                    eprintln!("{hash}: no constants captured for b{slot}: zeros");
                    Vec::new()
                });
                d.resize(64, 0);
                roots.push((k as u32 + 1, Root::View { srv: false, data: d }));
            }
            let uav_order: Vec<usize> = (0..n).filter(|&i| matches!(views[i], View::RawUav(_) | View::RawUavData(_) | View::TextureUav(..) | View::UintTextureUav(..))).collect();
            let out = host.dispatch(&root, &pso, &Dispatch { roots, views: &views, samplers: &[], groups }).unwrap();
            let Some(si) = scratch_at else {
                println!("{hash}: no scratch");
                continue;
            };
            let k = uav_order.iter().position(|&i| i == si).unwrap();
            let new = &out[k];
            let (mut changed, mut sat) = (0u64, 0u64);
            for (a, b) in scratch.iter().zip(new) {
                if a != b {
                    for s in 0..4 {
                        let (x, y) = ((a >> (8 * s)) as u8, (b >> (8 * s)) as u8);
                        if x != y {
                            changed += 1;
                            sat += u64::from(y == 0x7f || y == 0x80);
                        }
                    }
                }
            }
            println!("{hash} ({}x{}x{}): {changed} scratch bytes changed, {:.1}% saturated", groups[0], groups[1], groups[2], 100.0 * sat as f64 / changed.max(1) as f64);
            scratch.copy_from_slice(new);
            if let Some(dir) = std::env::var_os("VICTORY_REPLAY_DXIL") {
                let bytes: Vec<u8> = scratch.iter().flat_map(|w| w.to_le_bytes()).collect();
                std::fs::write(std::path::Path::new(&dir).join(format!("scratch-after-{hash}.bin")), bytes).unwrap();
            }
        }
    }

    /// Run victory30's constants for the upscale pass (b0, b1), frame 2400.
    const UPSCALE_B0: [u32; 64] = [0x3c0, 0x21c, 0x0, 0x0, 0x780, 0x438, 0x0, 0x0, 0x3c0, 0x21c, 0x0, 0x0, 0x3c0, 0x21c, 0x0, 0x0, 0x1e0, 0x10e, 0x0, 0x0, 0x1e0, 0x10e, 0x0, 0x0, 0x1e0, 0x10e, 0x0, 0x0, 0xf0, 0x87, 0x0, 0x0, 0xf0, 0x87, 0x0, 0x0, 0xf0, 0x87, 0x0, 0x0, 0x1e0, 0x10e, 0x0, 0x0, 0x1e0, 0x10e, 0x0, 0x0, 0x3c0, 0x21c, 0x0, 0x0, 0x3c0, 0x21c, 0x0, 0x0, 0x780, 0x438, 0x0, 0x0, 0x780, 0x438, 0x0, 0x0];
    const UPSCALE_B1: [u32; 64] = [0x3a088889, 0x3a72b9d6, 0x3fc00000, 0x3fc00000, 0x3f2aaaab, 0x3f2aaaab, 0xbd6fbf00, 0x3e25cd80, 0x3a4ccccd, 0x3ab60b61, 0x44f00000, 0x44870000, 0x44a00000, 0x44340000, 0x0, 0x0, 0x780, 0x438, 0x0, 0x500, 0x2d0, 0x3f800000, 0x3f800000, 0x0, 0x0, 0x0, 0x0, 0x0, 0xf0, 0x87, 0x0, 0x0, 0xf0, 0x87, 0x0, 0x0, 0xf0, 0x87, 0x0, 0x0, 0x1e0, 0x10e, 0x0, 0x0, 0x1e0, 0x10e, 0x0, 0x0, 0x3c0, 0x21c, 0x0, 0x0, 0x3c0, 0x21c, 0x0, 0x0, 0x780, 0x438, 0x0, 0x0, 0x780, 0x438, 0x0, 0x0];
}
