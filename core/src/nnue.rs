use candle_core::Device;
use candle_core::safetensors::SliceSafetensors;
use candle_nn::VarMap;

pub fn resolve_nnue() -> Result<nnue::Evaluator, Box<dyn std::error::Error>> {
    static NNUE_BYTES: &[u8] = include_bytes!("../../nnue/model.safetensors");

    let varmap = VarMap::new();
    let mut evaluator = nnue::Evaluator::new(&varmap, &Device::Cpu);
    let weights = SliceSafetensors::new(NNUE_BYTES)?;

    let mut tensors = varmap.data().lock().unwrap();
    for (name, var) in tensors.iter_mut() {
        let tensor = weights.load(name, var.device())?;
        var.set(&tensor)?;
    }

    evaluator.enable_nnue();

    Ok(evaluator)
}
