#include <ap_int.h>

using q_t = ap_int<64>;
using wide_t = ap_int<128>;

static const int FRAC_BITS = 40;
static const int MAX_STEPS = 8000;
static const int EVENT_BISECTIONS = 24;
static const q_t ONE_Q = q_t(1) << FRAC_BITS;
static const q_t STEP_Q = ONE_Q >> 10;
static const q_t ZS_Q = 24549305999887LL;

struct State {
    q_t x;
    q_t y;
    q_t w;
    q_t ell;
};

static q_t qmul(q_t a, q_t b) {
#pragma HLS INLINE
    wide_t product = wide_t(a) * wide_t(b);
    return q_t(product >> FRAC_BITS);
}

static q_t qdiv(q_t value, int divisor) {
#pragma HLS INLINE
    return value / divisor;
}

static State field(const State &state) {
#pragma HLS INLINE
    q_t xy = qmul(state.x, state.y);
    q_t half_w_zs = qdiv(state.w + ZS_Q, 2);
    State out;
    out.x = qmul(state.y, state.y) * 2 - xy;
    out.y = xy - qmul(state.y, half_w_zs);
    out.w = xy - state.w - ZS_Q;
    out.ell = state.x - state.y - half_w_zs - ONE_Q;
    return out;
}

static State add_scaled(const State &base, const State &delta, q_t scale) {
#pragma HLS INLINE
    State out;
    out.x = base.x + qmul(delta.x, scale);
    out.y = base.y + qmul(delta.y, scale);
    out.w = base.w + qmul(delta.w, scale);
    out.ell = base.ell + qmul(delta.ell, scale);
    return out;
}

static State rk4(const State &state, q_t step) {
#pragma HLS INLINE off
    State k1 = field(state);
    State k2 = field(add_scaled(state, k1, qdiv(step, 2)));
    State k3 = field(add_scaled(state, k2, qdiv(step, 2)));
    State k4 = field(add_scaled(state, k3, step));
    State weighted;
    weighted.x = k1.x + 2 * k2.x + 2 * k3.x + k4.x;
    weighted.y = k1.y + 2 * k2.y + 2 * k3.y + k4.y;
    weighted.w = k1.w + 2 * k2.w + 2 * k3.w + k4.w;
    weighted.ell = k1.ell + 2 * k2.ell + 2 * k3.ell + k4.ell;
    State out;
    out.x = state.x + qdiv(qmul(weighted.x, step), 6);
    out.y = state.y + qdiv(qmul(weighted.y, step), 6);
    out.w = state.w + qdiv(qmul(weighted.w, step), 6);
    out.ell = state.ell + qdiv(qmul(weighted.ell, step), 6);
    return out;
}

static void localize_event(const State &left, q_t &event_step, State &event_state) {
#pragma HLS INLINE off
    q_t low = 0;
    q_t high = STEP_Q;
    State high_state = rk4(left, high);
event_bisection:
    for (int iteration = 0; iteration < EVENT_BISECTIONS; ++iteration) {
#pragma HLS UNROLL factor=1
#pragma HLS LOOP_TRIPCOUNT min=24 max=24
        q_t middle = (low + high) >> 1;
        State middle_state = rk4(left, middle);
        if (middle_state.w < 0) {
            low = middle;
        } else {
            high = middle;
            high_state = middle_state;
        }
    }
    event_step = high;
    event_state = high_state;
}

extern "C" void target23_batch(const q_t *initial_xy, q_t *output, int n_leaves) {
#pragma HLS INTERFACE m_axi port=initial_xy bundle=gmem0 depth=662 max_read_burst_length=64
#pragma HLS INTERFACE m_axi port=output bundle=gmem1 depth=2648 max_write_burst_length=64
#pragma HLS INTERFACE s_axilite port=n_leaves
#pragma HLS INTERFACE s_axilite port=return

leaf_loop:
    for (int leaf = 0; leaf < n_leaves; ++leaf) {
#pragma HLS LOOP_TRIPCOUNT min=331 max=331
        State state = {initial_xy[2 * leaf], initial_xy[2 * leaf + 1], 0, 0};
        q_t time = 0;
        q_t event1_time = 0;
        q_t event2_time = 0;
        State event2_state = state;
        bool armed = false;
        int events = 0;
        int steps = 0;
step_loop:
        for (int step_index = 0; step_index < MAX_STEPS; ++step_index) {
#pragma HLS LOOP_TRIPCOUNT min=6500 max=7000
            State following = rk4(state, STEP_Q);
            if (following.w < 0) {
                armed = true;
            }
            if (armed && state.w < 0 && following.w >= 0) {
                q_t local_time;
                State localized;
                localize_event(state, local_time, localized);
                if (events == 0) {
                    event1_time = time + local_time;
                } else if (events == 1) {
                    event2_time = time + local_time;
                    event2_state = localized;
                }
                ++events;
                armed = false;
            }
            state = following;
            time += STEP_Q;
            steps = step_index + 1;
            if (events == 2) {
                break;
            }
        }
        q_t initial_normal = qmul(initial_xy[2 * leaf], initial_xy[2 * leaf + 1]) - ZS_Q;
        q_t final_normal = qmul(event2_state.x, event2_state.y) - ZS_Q;
        q_t flags = 0;
        if (events == 2) flags |= q_t(1);
        if (initial_normal > 0) flags |= q_t(2);
        if (final_normal > 0) flags |= q_t(4);
        const int base = 8 * leaf;
        output[base + 0] = steps;
        output[base + 1] = events;
        output[base + 2] = event1_time;
        output[base + 3] = event2_time;
        output[base + 4] = event2_state.x;
        output[base + 5] = event2_state.y;
        output[base + 6] = event2_state.ell;
        output[base + 7] = flags;
    }
}
