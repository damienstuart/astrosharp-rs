//! Inference for the `neuralnet` multilayer perceptrons used by AstroSharp.
//!
//! Mirrors `neuralnet:::predict.nn`: each layer computes `act(cbind(1, x) %*% W)`,
//! where the first row of `W` holds the biases. Hidden layers use `act.fct`, the
//! output layer uses `output.act.fct` (`linear.output = FALSE` for every model).

use crate::Error;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Activation {
    Relu,
    Logistic,
}

impl Activation {
    fn from_code(code: u32) -> Result<Self, Error> {
        match code {
            0 => Ok(Activation::Relu),
            1 => Ok(Activation::Logistic),
            _ => Err(Error::BadWeights(format!("unknown activation code {code}"))),
        }
    }

    fn apply(self, xs: &mut [f64]) {
        match self {
            Activation::Relu => xs.iter_mut().for_each(|x| {
                if *x <= 0.0 {
                    *x = 0.0;
                }
            }),
            Activation::Logistic => xs.iter_mut().for_each(|x| *x = 1.0 / (1.0 + (-*x).exp())),
        }
    }
}

#[derive(Clone, Debug)]
pub struct Layer {
    pub inputs: usize,
    pub outputs: usize,
    pub bias: Vec<f64>,
    /// `inputs x outputs`, row-major.
    pub weights: Vec<f64>,
}

#[derive(Clone, Debug)]
pub struct Mlp {
    pub layers: Vec<Layer>,
    pub hidden: Activation,
    pub output: Activation,
}

/// Reusable buffers for [`Mlp::forward`].
#[derive(Default)]
pub struct Scratch {
    a: Vec<f64>,
    b: Vec<f64>,
}

impl Mlp {
    /// Parses the format written by `tools/export_weights.R`.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, Error> {
        let bad = |msg: &str| Error::BadWeights(msg.to_string());
        if bytes.len() < 20 || &bytes[..4] != b"ASNN" {
            return Err(bad("missing ASNN header"));
        }
        let mut pos = 4;
        let u32_at = |pos: &mut usize| -> Result<u32, Error> {
            let v = bytes
                .get(*pos..*pos + 4)
                .ok_or_else(|| bad("truncated header"))?;
            *pos += 4;
            Ok(u32::from_le_bytes(v.try_into().unwrap()))
        };
        if u32_at(&mut pos)? != 1 {
            return Err(bad("unsupported version"));
        }
        let hidden = Activation::from_code(u32_at(&mut pos)?)?;
        let output = Activation::from_code(u32_at(&mut pos)?)?;
        let n_layers = u32_at(&mut pos)? as usize;
        let mut dims = Vec::with_capacity(n_layers);
        for _ in 0..n_layers {
            let rows = u32_at(&mut pos)? as usize;
            let cols = u32_at(&mut pos)? as usize;
            dims.push((rows, cols));
        }
        let mut layers = Vec::with_capacity(n_layers);
        for (i, &(rows, cols)) in dims.iter().enumerate() {
            if rows < 2 || (i > 0 && rows != dims[i - 1].1 + 1) {
                return Err(bad("inconsistent layer dimensions"));
            }
            let n = rows * cols;
            let raw = bytes
                .get(pos..pos + n * 8)
                .ok_or_else(|| bad("truncated weights"))?;
            pos += n * 8;
            let vals: Vec<f64> = raw
                .as_chunks::<8>()
                .0
                .iter()
                .map(|c| f64::from_le_bytes(*c))
                .collect();
            layers.push(Layer {
                inputs: rows - 1,
                outputs: cols,
                bias: vals[..cols].to_vec(),
                weights: vals[cols..].to_vec(),
            });
        }
        if pos != bytes.len() {
            return Err(bad("trailing bytes"));
        }
        if layers.last().map(|l| l.outputs) != Some(1) {
            return Err(bad("expected a single output"));
        }
        Ok(Mlp {
            layers,
            hidden,
            output,
        })
    }

    pub fn inputs(&self) -> usize {
        self.layers[0].inputs
    }

    /// Evaluates `n` samples. `input` is `n x inputs()` row-major; `out` receives `n` values.
    pub fn forward(&self, input: &[f64], n: usize, scratch: &mut Scratch, out: &mut [f64]) {
        assert_eq!(input.len(), n * self.inputs());
        assert_eq!(out.len(), n);
        let mut prev = std::mem::take(&mut scratch.a);
        let mut next = std::mem::take(&mut scratch.b);
        let last = self.layers.len() - 1;
        for (i, layer) in self.layers.iter().enumerate() {
            let src: &[f64] = if i == 0 { input } else { &prev };
            next.clear();
            for _ in 0..n {
                next.extend_from_slice(&layer.bias);
            }
            // SAFETY: `src` is n x inputs, `weights` is inputs x outputs and `next` is
            // n x outputs, all row-major and contiguous, matching the strides below.
            unsafe {
                matrixmultiply::dgemm(
                    n,
                    layer.inputs,
                    layer.outputs,
                    1.0,
                    src.as_ptr(),
                    layer.inputs as isize,
                    1,
                    layer.weights.as_ptr(),
                    layer.outputs as isize,
                    1,
                    1.0,
                    next.as_mut_ptr(),
                    layer.outputs as isize,
                    1,
                );
            }
            if i == last {
                self.output.apply(&mut next);
            } else {
                self.hidden.apply(&mut next);
            }
            std::mem::swap(&mut prev, &mut next);
        }
        out.copy_from_slice(&prev);
        scratch.a = prev;
        scratch.b = next;
    }
}
