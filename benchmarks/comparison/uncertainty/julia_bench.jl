# Benchmark: PK Uncertainty Propagation — Julia equivalent
# Sobol-style variance decomposition for PK parameter sensitivity.

struct PKParameter
    code::Int
    value::Float64
    std_uncert::Float64
    sensitivity::Float64
    contribution::Float64
end

function make_pk_param(code, value, std_uncert, sensitivity)
    contribution = sensitivity^2 * std_uncert^2
    PKParameter(code, value, std_uncert, sensitivity, contribution)
end

function is_clearance_related(code)
    code in (0, 3, 6, 7)
end

function is_volume_related(code)
    code in (1, 6)
end

function is_absorption_related(code)
    code in (2, 4)
end

function analyze_pk(params::Vector{PKParameter})
    total_var = sum(p.contribution for p in params)
    cl_var = sum(p.contribution for p in params if is_clearance_related(p.code))
    vol_var = sum(p.contribution for p in params if is_volume_related(p.code))
    abs_var = sum(p.contribution for p in params if is_absorption_related(p.code))

    cl_frac = total_var > 0 ? cl_var / total_var : 0.0
    vol_frac = total_var > 0 ? vol_var / total_var : 0.0
    abs_frac = total_var > 0 ? abs_var / total_var : 0.0

    dominant_idx = argmax([p.contribution for p in params])
    is_cl_dominant = cl_frac > 0.7

    return (cl_frac=cl_frac, vol_frac=vol_frac, abs_frac=abs_frac,
            dominant_idx=dominant_idx, is_cl_dominant=is_cl_dominant,
            total_var=total_var)
end

function main()
    params = [
        make_pk_param(0, 5.0, 1.5, 0.9),   # CL
        make_pk_param(1, 50.0, 5.0, 0.2),   # V
        make_pk_param(2, 1.2, 0.3, 0.15),   # ka
        make_pk_param(3, 0.8, 0.05, 0.1),   # F
        make_pk_param(4, 0.5, 0.1, 0.05),   # t_lag
        make_pk_param(7, 1.2, 0.15, 0.3),   # SCr
        make_pk_param(6, 70.0, 10.0, 0.12), # Weight
        make_pk_param(5, 0.0, 0.0, 0.0),    # Other
    ]

    n_iters = 10_000

    # Warmup
    analyze_pk(params)

    local result
    t = @elapsed begin
        for _ in 1:n_iters
            result = analyze_pk(params)
        end
    end

    println("PK uncertainty ($n_iters iters): CL contribution = $(result.cl_frac), time = $(t*1000) ms")
    if result.is_cl_dominant
        println("Recommendation: TDM (clearance dominant)")
    end
end

main()
