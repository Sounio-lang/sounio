# Benchmark: QR Decomposition — Julia equivalent
# Hand-rolled Householder QR on 4x4 matrices (no LinearAlgebra.qr) for fair comparison.

function mat_get(data, rows, cols, r, c)
    return data[r * cols + c + 1]  # Julia is 1-indexed
end

function mat_set!(data, rows, cols, r, c, val)
    data[r * cols + c + 1] = val
end

function qr_householder!(A::Matrix{Float64})
    n, m = size(A)
    Q = Matrix{Float64}(I, n, n)
    R = copy(A)

    for k in 1:min(n, m)
        # Extract column
        col = R[k:n, k]
        alpha = -sign(col[1]) * norm(col)

        # Householder vector
        v = copy(col)
        v[1] -= alpha
        v_norm = norm(v)
        if v_norm > 1e-14
            v ./= v_norm

            # Apply to R
            R[k:n, :] .-= 2.0 .* v * (v' * R[k:n, :])

            # Apply to Q
            Q[:, k:n] .-= 2.0 .* (Q[:, k:n] * v) * v'
        end
    end
    return Q, R
end

function main()
    using LinearAlgebra

    A = [4.0 1.0 2.0 1.0;
         1.0 3.0 1.0 2.0;
         2.0 1.0 5.0 1.0;
         1.0 2.0 1.0 4.0]

    n_iters = 1000

    # Warmup
    qr_householder!(copy(A))

    max_err = 0.0
    t = @elapsed begin
        for _ in 1:n_iters
            Q, R = qr_householder!(copy(A))
            err = norm(A - Q * R)
            max_err = max(max_err, err)
        end
    end

    println("QR 4x4 ($n_iters iters): max ||A-QR|| = $max_err, time = $(t*1000) ms")
end

main()
