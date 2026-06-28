struct PbpkInput {
  time_h: f32,
  ka_xr: f32,
  f_oral: f32,
  cl_cyp2d6: f32,
  cl_odv: f32,
  phenotype_scale: f32,
  parent_cmax_ng_ml: f32,
  odv_cmax_ng_ml: f32,
};

struct PbpkOutput {
  release_fraction: f32,
  parent_ng_ml: f32,
  odv_ng_ml: f32,
  odv_parent_ratio: f32,
};

@group(0) @binding(0) var<storage, read> input_state: array<PbpkInput>;
@group(0) @binding(1) var<storage, read_write> output_state: array<PbpkOutput>;

fn xr_release(time_h: f32, ka_xr: f32) -> f32 {
  let centered = (time_h - 5.5) / max(ka_xr, 0.01);
  return exp(-(centered * centered));
}

fn bounded(value: f32, low: f32, high: f32) -> f32 {
  return min(max(value, low), high);
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
  let index = global_id.x;
  if (index >= arrayLength(&input_state)) {
    return;
  }

  let x = input_state[index];
  let release = xr_release(x.time_h, x.ka_xr) * bounded(x.f_oral, 0.0, 1.0);
  let parent_clearance = exp(-x.cl_cyp2d6 * max(x.time_h - 5.5, 0.0));
  let odv_clearance = exp(-x.cl_odv * max(x.time_h - 9.0, 0.0));
  let conversion = bounded(x.phenotype_scale, 0.1, 2.0);

  let parent = x.parent_cmax_ng_ml * release * parent_clearance / conversion;
  let odv = x.odv_cmax_ng_ml * release * (1.0 - exp(-0.18 * x.time_h * conversion)) * odv_clearance;
  let ratio = odv / max(parent, 0.001);

  output_state[index] = PbpkOutput(
    bounded(release, 0.0, 1.0),
    max(parent, 0.0),
    max(odv, 0.0),
    max(ratio, 0.0),
  );
}
