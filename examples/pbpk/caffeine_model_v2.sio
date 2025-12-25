struct Drug {
    mw: f64,
    logp: f64,
    fu: f64,
    pka: f64
}

struct PBPKParams {
    cl: f64@L_per_h,
    vd: f64@L,
    ka: f64@per_h,
    bioavail: f64
}

struct PKResults {
    cmax: f64@mg_per_L,
    tmax: f64@h,
    auc: f64@mg_h_per_L,
    t_half: f64@h
}

fn calc_kp_tissue(fu: f64, tissue_lipid: f64, tissue_water: f64) -> f64 {
    let plasma_lipid = 0.007;
    let plasma_water = 0.94;
    let p_ow = 0.85;
    
    let numerator = tissue_lipid * p_ow + tissue_water;
    let denominator = plasma_lipid * p_ow + plasma_water;
    let kp = (numerator / denominator) * fu;
    
    return kp
}

fn calc_pk_metrics(dose: f64@mg, params: PBPKParams) -> PKResults {
    let dose_absorbed = dose * params.bioavail;
    let ke = params.cl / params.vd;
    let t_half = 0.693 / ke;
    let tmax = 0.8;
    let cmax = (dose_absorbed / params.vd) * 0.7;
    let auc = dose_absorbed / params.cl;
    
    return PKResults {
        cmax: cmax,
        tmax: tmax,
        auc: auc,
        t_half: t_half
    }
}

fn main() -> i32 {
    let drug = Drug {
        mw: 194.19,
        logp: -0.07,
        fu: 0.65,
        pka: 10.4
    };
    
    let params = PBPKParams {
        cl: 6.0,
        vd: 40.0,
        ka: 4.5,
        bioavail: 0.97
    };
    
    let dose = 200.0;
    
    let kp_liver = calc_kp_tissue(drug.fu, 0.04, 0.72);
    let kp_kidney = calc_kp_tissue(drug.fu, 0.02, 0.77);
    let kp_muscle = calc_kp_tissue(drug.fu, 0.01, 0.76);
    let kp_adipose = calc_kp_tissue(drug.fu, 0.85, 0.14);
    let kp_brain = calc_kp_tissue(drug.fu, 0.05, 0.77) * 1.2;
    
    let results = calc_pk_metrics(dose, params);
    
    let cmax_obs = 4.5;
    let auc_obs = 30.0;
    let t_half_obs = 5.0;
    
    let cmax_ratio = results.cmax / cmax_obs;
    let auc_ratio = results.auc / auc_obs;
    let t_half_ratio = results.t_half / t_half_obs;
    
    let cmax_pass = cmax_ratio >= 0.5 && cmax_ratio <= 2.0;
    let auc_pass = auc_ratio >= 0.5 && auc_ratio <= 2.0;
    let t_half_pass = t_half_ratio >= 0.5 && t_half_ratio <= 2.0;
    
    let cl_rapid = params.cl * 1.5;
    let cl_slow = params.cl * 0.5;
    let t_half_rapid = 0.693 * params.vd / cl_rapid;
    let t_half_slow = 0.693 * params.vd / cl_slow;
    
    let cl_inhibited = params.cl * 0.2;
    let auc_inhibited = (dose * params.bioavail) / cl_inhibited;
    let auc_fold_increase = auc_inhibited / results.auc;
    
    let all_pass = cmax_pass && auc_pass && t_half_pass;
    
    println("=== CAFFEINE PBPK ===")
    println("Dose: 200 mg oral")
    println("Cmax pred (mg/L):")
    println(results.cmax)
    println("AUC pred (mg*h/L):")
    println(results.auc)
    println("t1/2 pred (h):")
    println(results.t_half)
    println("Kp brain:")
    println(kp_brain)
    println("DDI AUC fold increase:")
    println(auc_fold_increase)
    return 0
}
