//! The vertex shader that draws a mesh program run as compute
//! ([`super::translate_mesh_as_compute`]) on hosts without mesh shaders.
//! The compute shader leaves each group's vertices in the root UAV
//! `SLOT_MESH_VERTICES` (`vertex_dwords` dwords a vertex, dword `row * 4 +
//! column` of the vertex signature) and their triangles as indices; an
//! indexed draw over those indices runs this shader, which reads vertex
//! `SV_VertexID` from the same buffer (bound as the root SRV
//! `t<SLOT_MESH_VERTICES>` in `ROOT_CBV_SPACE`) and outputs it with the
//! mesh program's vertex signature, register for register, so it links with
//! the title's pixel shader.

use super::bitcode::{Attr, Bin, Cast, Term};
use super::container::SigElement;
use super::shader::{SLOT_MESH_INDICES, SLOT_MESH_VERTICES, Shader};
use crate::ShaderError;
use crate::dxbc::{Container, FourCC};

/// The vertex shader (a DXIL container) for the mesh program whose mesh
/// translation is `ms` (its [`super::HostShader::dxil`], for the vertex
/// signature) and whose compute translation has `vertex_dwords` dwords a
/// vertex ([`super::MeshAsCompute::vertex_dwords`]).
pub fn mesh_vertex_shader(ms: &[u8], vertex_dwords: u32) -> Result<Vec<u8>, ShaderError> {
    let c = Container::parse(ms)?;
    let outputs = match c.part(FourCC::OSG1) {
        Some(part) => super::signature(part.data)?,
        None => Vec::new(),
    };
    build(outputs, vertex_dwords)
}

fn build(mut outputs: Vec<SigElement>, vertex_dwords: u32) -> Result<Vec<u8>, ShaderError> {
    let unsupported = |what| ShaderError::Unsupported { offset: 0, what };
    if outputs.iter().any(|e| e.row.is_none()) {
        return Err(unsupported("a mesh vertex output without a register"));
    }
    if super::lower::mesh_vertex_dwords(&outputs) != vertex_dwords {
        return Err(unsupported("a mesh vertex signature unlike its compute translation's"));
    }
    for e in &mut outputs {
        // Interpolation as a vertex shader's outputs are declared (`rect.rs`).
        e.interp = match (e.kind, e.comp) {
            (3, _) => 4,
            (0 | 6 | 7, 9) => 2,
            (0, _) => 1,
            _ => 0,
        };
        e.usage = (1 << e.cols) - 1;
    }
    let mut s = Shader::new(false);
    s.inputs = vec![SigElement {
        name: "SV_VertexID".into(),
        index: 0,
        kind: 1,
        comp: 5,
        interp: 0,
        row: Some(0),
        col: 0,
        cols: 1,
        usage: 1,
        half: false,
    }];
    s.outputs = outputs.clone();
    let b = s.entry;
    let (i32t, res, void) = (s.t.i32, s.t.res_ret, s.t.void);
    let iu = s.m.undef(i32t);
    let args = [s.m.i32(0), s.m.i32(0), s.m.i8(0), iu];
    let index = s.op(b, "loadInput.i32", Attr::ReadNone, i32t, 4, &args);
    // A plain (not indexed) draw: vertex n of the draw is index n of the
    // compute shader's index buffer, which names the vertex. Both buffers
    // stay unordered-access views (root UAVs), so nothing changes state
    // between the compute shader and the draw.
    let indices = s.root_uav_handle(SLOT_MESH_INDICES);
    let k4 = s.m.i32(4);
    let at = s.m.bin(b, Bin::Mul, index, k4);
    let (one, align) = (s.m.i8(1), s.m.i32(4));
    let r = s.op(b, "rawBufferLoad.f32", Attr::ReadOnly, res, 139, &[indices, at, iu, one, align]);
    let r = s.m.extract(b, r, 0);
    let vertex = s.m.cast(b, Cast::Bitcast, r, i32t);
    let stride = s.m.i32(vertex_dwords * 4);
    let base = s.m.bin(b, Bin::Mul, vertex, stride);
    let vertices = s.root_uav_handle(SLOT_MESH_VERTICES);
    for (id, e) in outputs.iter().enumerate() {
        let row = e.row.expect("checked above");
        for col in 0..e.cols {
            let dword = row * 4 + u32::from(e.col) + u32::from(col);
            let offset = s.m.i32(dword * 4);
            let at = s.m.bin(b, Bin::Add, base, offset);
            let (one, align) = (s.m.i8(1), s.m.i32(4));
            let r = s.op(b, "rawBufferLoad.f32", Attr::ReadOnly, res, 139, &[vertices, at, iu, one, align]);
            let f = s.m.extract(b, r, 0);
            let (name, v) = if e.comp == 9 {
                ("storeOutput.f32", f)
            } else {
                ("storeOutput.i32", s.m.cast(b, Cast::Bitcast, f, i32t))
            };
            let args = [s.m.i32(id as u32), s.m.i32(0), s.m.i8(col), v];
            s.op(b, name, Attr::NoUnwind, void, 5, &args);
        }
    }
    s.m.terminate(b, Term::Ret);
    Ok(s.finish())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn element(name: &str, index: u32, kind: u8, comp: u8, row: u32, cols: u8) -> SigElement {
        SigElement {
            name: name.into(),
            index,
            kind,
            comp,
            interp: 0,
            row: Some(row),
            col: 0,
            cols,
            usage: 0,
            half: false,
        }
    }

    /// A position, a float interpolant and an integer one: a vertex shader
    /// with those outputs and one vertex-id input, which the validator,
    /// where there is one, accepts; a layout unlike the outputs' is refused.
    #[test]
    fn the_mesh_vertex_stage_is_a_vertex_shader() {
        let outputs = vec![
            element("SV_Position", 0, 3, 9, 0, 4),
            element("TEXCOORD", 0, 0, 9, 1, 4),
            element("TEXCOORD", 1, 0, 5, 2, 2),
        ];
        let vs = build(outputs.clone(), 12).unwrap();
        let c = Container::parse(&vs).unwrap();
        // Program version: vertex (1), shader model 6.6.
        assert_eq!(c.part(FourCC::DXIL).unwrap().data[..4], [0x66, 0, 1, 0]);
        let sig = |fourcc| super::super::signature(c.part(fourcc).unwrap().data).unwrap();
        let (ins, outs) = (sig(FourCC::ISG1), sig(FourCC::OSG1));
        assert_eq!((ins.len(), ins[0].kind), (1, 1));
        assert_eq!(outs.len(), 3);
        for (a, b) in outs.iter().zip(&outputs) {
            assert_eq!((&a.name, a.index, a.row, a.cols, a.comp), (&b.name, b.index, b.row, b.cols, b.comp));
        }
        assert!(build(outputs, 8).is_err());
        #[cfg(windows)]
        if let Ok(mut v) = super::super::validator::Validator::load(
            super::super::validator::Validator::default_path().as_deref(),
        ) {
            v.validate(&mut vs.clone()).unwrap();
        }
    }
}
