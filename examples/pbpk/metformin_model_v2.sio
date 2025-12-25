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

fn calc_kp_tissue(fu: f64, uptake_factor: f64) -> f64 {
    let kp = 0.745 * fu * uptake_factor;
    return kp
}

fn calc_pk_metrics(dose: f64@mg, params: PBPKParams) -> PKResults {
    let dose_absorbed = dose * params.bioavail;
    let ke = params.cl / params.vd;
    let t_half = 0.693 / ke;
    let tmax = 2.5;
    let cmax = (dose_absorbed / params.vd) * 0.6;
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
        mw: 129.16,
        logp: -1.43,
        fu: 1.0,
        pka: 12.4
    };
    
    let params = PBPKParams {
        cl: 350.0,
        vd: 400.0,
        ka: 0.9,
        bioavail: 0.5
    };
    
    let dose = 500.0;
    
    let kp_liver = calc_kp_tissue(drug.fu, 3.5);
    let kp_kidney = calc_kp_tissue(drug.fu, 5.0);
    let kp_muscle = calc_kp_tissue(drug.fu, 1.0);
    let kp_brain = calc_kp_tissue(drug.fu, 0.1);
    
    let results = calc_pk_metrics(dose, params);
    
    let cmax_obs = 1.2;
    let auc_obs = 5.5;
    let t_half_obs = 5.0;
    
    let cmax_ratio = results.cmax / cmax_obs;
    let auc_ratio = results.auc / auc_obs;
    let t_half_ratio = results.t_half / t_half_obs;
    
    let cmax_pass = cmax_ratio >= 0.5 && cmax_ratio <= 2.0;
    let auc_pass = auc_ratio >= 0.5 && auc_ratio <= 2.0;
    let t_half_pass = t_half_ratio >= 0.5 && t_half_ratio <= 2.0;
    
    let all_pass = cmax_pass && auc_pass && t_half_pass;
    
    println("=== METFORMIN PBPK ===")
    println("Dose: 500 mg oral")
    println("Cmax pred (mg/L):")
    println(results.cmax)
    println("AUC pred (mg*h/L):")
    println(results.auc)
    println("t1/2 pred (h):")
    println(results.t_half)
    println("Kp liver:")
    println(kp_liver)
    println("Kp kidney:")
    println(kp_kidney)
    return 0
}
