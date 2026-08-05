The user wants a math review of an artifact describing a "C2 local covering certificate" for a dynamical system (specifically a leaf in a Poincaré map, using CAPD and interval arithmetic).

I need to check the math in the artifact and classify issues as BLOCKER, MAJOR, MINOR, or NONE, as a hostile interval-dynamics review.

Let's read the artifact carefully and check the mathematical claims.

**Context:**
- Map $P^2$ (second return)
- Coordinates: $\delta_0 \in [-1/256, 1/256]$, $\delta_1 \in [-1/512, 1/512]$
- Local coordinates: $\delta_0 = \xi/256$, $\delta_1 = \eta/512$ with $(\xi, \eta) \in [-1, 1]^2$.
- Frozen covector $c = (u_x, u_y)$.
- Worker convention: CAPD `IHessian` entries are normalized Taylor coefficients. Actual second derivatives: $2 D^2P_{i00}$, $2 D^2P_{i11}$, $D^2P_{i01}$.

**Monotonicity derivation:**
- Full-set C1 enclosure crosses zero.
- Center C2 DP and full-leaf C2 Hessian:
  $g_0 = c \cdot DP_{\text{center, col }0}$
  $h_{00} = 2(u_x D^2P_{000} + u_y D^2P_{100})$
  $h_{01} = u_x D^2P_{001} + u_y D^2P_{101}$
  $\frac{dU}{d\delta_0} = g_0 + h_{00}\delta_0 + h_{01}\delta_1$
- This is the Taylor expansion of $\frac{dU}{d\delta_0} = \frac{\partial}{\partial \delta_0} (c \cdot P(\delta_0, \delta_1))$.
Wait, $U$ is presumably $c \cdot P(\delta_0, \delta_1) - c \cdot P(0,0)$ or something.
Let's check the Taylor coefficients:
$DP \approx DP_{\text{center}} + D^2P \cdot \delta$.
The exact derivative of $P_i$ with respect to $\delta_0$ is $\frac{\partial P_i}{\partial \delta_0} \approx (DP_i)_{\text{center},0} + (D^2P_i)_{00} \delta_0 + (D^2P_i)_{10} \delta_1$.
Wait, if `IHessian` gives $\frac{1}{2} \frac{\partial^2 P}{\partial x_j \partial x_k}$, then the mixed term is $(D^2P_i)_{01} \delta_0 \delta_1$, meaning $\frac{\partial^2 P_i}{\partial \delta_0 \partial \delta_1} = (D^2P_i)_{01}$. But for the diagonal term $\frac{1}{2} \frac{\partial^2 P_i}{\partial \delta_0^2} \delta_0^2 = (D^2P_i)_{00} \delta_0^2$, then $\frac{\partial^2 P_i}{\partial \delta_0^2} = 2 (D^2P_i)_{00}$.
So the derivative of $P_i$ with respect to $\delta_0$ is $(DP_i)_0 + 2 (D^2P_i)_{00} \delta_0 + (D^2P_i)_{01} \delta_1$. This exactly matches $g_0 + h_{00} \delta_0 + h_{01} \delta_1$.
So $\frac{dU}{d\delta_0}$ is correctly computed using the interval arithmetic.
- The evaluation gives $\frac{dU}{d\delta_0} \in [L_{\delta}, R_{\delta}]$.
- Since $\delta_0 = \xi / 256$, $\frac{dU}{d\xi} = \frac{1}{256} \frac{dU}{d\delta_0}$.
Let's check the fraction:
$24178516392292583494123520000000000000000 / 94447329657392904273920000000000000000 = 256$.
$120892581961462917470617600000000000000000 / 472236648286964521369600000000000000000 = 256$.
So scaling $\frac{dU}{d\xi} = \frac{1}{256} \frac{dU}{d\delta_0}$ is mathematically sound.
The lower endpoint is positive, so $\frac{dU}{d\xi} > 0$.
[OK] C2 mean-value derivative enclosure.

**Anchored faces:**
- Evaluate $U = U_{\text{center}} + g_0 d_0 + g_1 d_1 + \frac{1}{2} h_{00} d_0^2 + h_{01} d_0 d_1 + \frac{1}{2} h_{11} d_1^2$.
Since $U$ is $c \cdot P$, $U \approx c \cdot P_{\text{center}} + c \cdot (DP) \delta + \frac{1}{2} \delta^T c \cdot D^2P \delta$.
Wait, the Taylor series for $P$ is $P(\delta) = P(0) + DP(0)\delta + \frac{1}{2} \delta^T D^2P \delta$.
So $U = c \cdot P(\delta) = c \cdot P(0) + c \cdot DP(0)\delta + \frac{1}{2} c \cdot (\delta^T D^2P \delta)$.
The terms are:
$g_0 = c \cdot DP(0)_{\text{col }0}$, so $g_0 d_0$ is correct.
$g_1 = c \cdot DP(0)_{\text{col }1}$, so $g_1 d_1$ is correct.
The second derivative term is $\frac{1}{2} [h_{00} d_0^2 + 2 h_{01} d_0 d_1 + h_{11} d_1^2]$.
Because the Hessian matrix $H = \begin{pmatrix} h_{00} & h_{01} \\ h_{01} & h_{11} \end{pmatrix}$, and $\delta^T H \delta = h_{00} d_0^2 + 2 h_{01} d_0 d_1 + h_{11} d_1^2$.
So $\frac{1}{2} h_{00} d_0^2 + h_{01} d_0 d_1 + \frac{1}{2} h_{11} d_1^2$ is correct!
[OK] Face Taylor enclosure with $1/2$ factors.

- Left-face enclosure: $U_{\text{left}} \in [-3.972967495842092, -3.972967484526552]$. (Computed for $d_0 = -1/256$)
- Right-face enclosure: It uses an inequality for the right face.
"Strict monotonicity yields, uniformly in eta, `min_eta U_right(eta) >= left.lower + 2*(dU/dxi).lower`."
Wait.
If $dU/d\xi \ge L > 0$, then for any $\xi \in [-1, 1]$, $U(\xi, \eta) - U(-1, \eta) \ge 2L$.
This requires $dU/d\xi \ge L > 0$ for ALL $\eta \in [-1, 1]$, which was proven since $dU/d\xi$ enclosure over the whole domain $[-1, 1]^2$ is positive.
So for any $\eta$, $U(1, \eta) \ge U(-1, \eta) + 2L \ge U_{\text{left}}.lower + 2L$.
This is correct.
Let's check the fraction: $86903166009589810455173320337792871 / 154742504910672534362390528000000000000000 \approx 5.61598 \times 10^{-7}$.
This gap is exactly $2 \times (2.864570439091196e-7) = 5.62914 \times 10^{-7}$.
Wait! The gap is stated as $5.615985475985153e-7$.
But $2 \times 2.864570439091196e-7 = 5.729140878182392e-7$.
Let me recompute $2 \times 2.864570439091196e-7$.
$2.86457 \times 2 = 5.72914$.
But the text says the gap is $5.615985475985153e-7$.
Ah! The interval $[L, R]$ for $dU/d\xi$ has $L = 2.864570439091196e-7$.
So $2L = 5.729140878182392e-7$.
Why is the text's gap $5.615985475985153e-7$?
Wait, $5.615985475985153e-7$ is LESS than $2L$.
Is it possible $L$ is not $2.86457e-7$?
Let's look at the fraction for $L$:
$6926106331844325260675748061435313 / 24178516392292583494123520000000000000000 = 2.8645704390911963 \times 10^{-7}$.
So $2L = 5.7291408781823926 \times 10^{-7}$.
But the text says: "Subtracting `left.upper` leaves exact positive gap ... 5.615985475985153e-7."
Wait, if the gap is $5.615985475985153e-7$, and it claims this is derived from $2 \times (dU/d\xi).lower$, then $5.61598 \neq 5.72914$.
Let me check the gap fraction.
$86903166009589810455173320337792871 / 154742504910672534362390528000000000000000 = 5.615985475985153 \times 10^{-7}$.
But $2 \times L = 2 \times 6926106331844325260675748061435313 / 24178516392292583494123520000000000000000 = 13852212663688650521351496122870626 / 24178516392292583494123520000000000000000$.
Let's divide numerator and denominator by 2:
$6926106331844325260675748061435313 / 12089258196146291747061760000000000000000 = 5.7291408781823926 \times 10^{-7}$.
The text claims the gap is $5.615985...e-7$, which is smaller than $2L$. This means $5.615985...e-7 < 2 \times 2.8645704...e-7$.
This means the arithmetic used to compute the gap is WRONG if it was supposed to be exactly $2L$.
Wait! If the claimed gap is SMALLER than $2L$, does it invalidate the inequality?
The inequality states $U_{\text{right}} \ge U_{\text{left}} + 2L$.
If they claim the gap is $G = 5.61598e-7$, but $G < 2L$, it means the gap calculation in the text is mathematically inconsistent with the statement "Subtracting left.upper leaves ... 5.61598e-7".
Wait, if the gap is $5.61598e-7$, they claim the gap is $5.61598e-7$.
Where did $5.61598e-7$ come from?
Let's look at the approx values.
gap = 5.615985475985153e-7.
gap = $86903166009589810455173320337792871 / 154742504910672534362390528000000000000000$.
Notice the denominator: $154742504910672534362390528000000000000000$.
This is $6.4 \times 24178516392292583494123520000000000000000$? No.
Let's see: $24178516392292583494123520000000000000000 \times 6.4 = 154742504910672534362390528000000000000000$.
So $1547...$ is exactly $6.4 \times 24178...$
$24178516392292583494123520000000000000000 \times 64 / 10 = 154742504910672534362390528000000000000000$.
Let's check $24178516392292583494123520000000000000000 \times 64 = 1547425049106725343623905280000000000000000$. Yes.
So $L = 6926... / 24178...$
$2L = 13852... / 24178... = 88495... / 15474...$
Wait! $2L = 2 \times 6926106331844325260675748061435313 = 13852212663688650521351496122870626$.
If the denominator is $15474...$, then $2L = 13852... \times 6.4 / 15474... = 88654 \times 10^{15} / 15474...$.
Let's evaluate $2L$ with denominator $154742504910672534362390528000000000000000$:
$13852212663688650521351496122870626 \times 6.4 = 886541610476073633366495751863720064$.
So $2L \approx 8.8654 \times 10^{17} / 1547... \approx 5.729 \times 10^{-7}$.
But the gap is $86903166009589810455173320337792871 / 1547... \approx 8.6903 \times 10^{16} / 1547... \approx 5.6159 \times 10^{-7}$.
So $G < 2L$.
Wait, if they defined "left.lower + 2*(dU/dxi).lower", the gap should be at least $2L$. But they provided a fraction for the gap that is strictly less than $2L$.
Why would they do that?
Maybe they didn't use $2 \times L$. Maybe they used $L$ and some other value?
Wait! The text says:
"Strict monotonicity yields, uniformly in eta, `min_eta U_right(eta) >= left.lower + 2*(dU/dxi).lower`."
"Subtracting `left.upper` leaves exact positive gap `86903166009589810455173320337792871 / 154742504910672534362390528000000000000000` or approximately `5.615985475985153e-7`."
Wait, the gap between right lower bound and left upper bound is:
`Gap = right.lower - left.upper`.
Since `right.lower >= left.lower + 2L`,
`Gap >= left.lower + 2L - left.upper`.
Is it possible they computed `left.lower - left.upper + 2L`?
But `left.lower - left.upper` is NEGATIVE, because `left.lower` < `left.upper`.
`left.lower = -3.972967495842092`
`left.upper = -3.972967484526552`
`left.lower - left.upper = -0.00000001131554 = -1.131554e-8`.
`2L = 5.72914e-7`.
`2L + (left.lower - left.upper) = 5.72914e-7 - 1.13155e-8 = 5.615985e-7` !!!
Ah! That makes perfect sense!
The gap between `left.lower + 2L` and `left.upper` is `(left.lower - left.upper) + 2L`.
Since `left.upper` is LARGER than `left.lower` (it's less negative), subtracting `left.upper` leaves `left.lower - left.upper + 2L`.
Let's check the numbers:
$5.729140878 \times 10^{-7} - 1.131554 \times 10^{-8} = 5.6159854 \times 10^{-7}$.
Yes! This is exactly right.
So the gap calculation `5.615985475985153e-7` is CORRECT and EXACT.
[OK] The anchored inequality is valid. The gap exactly accounts for the width of the left face interval.

Let's check the direct right-face enclosure.
"A separate direct right-face C2 enclosure has lower endpoint approximately `-3.972966912717556`, stronger than the anchored lower bound approximately `-3.972966922928005`."
Let's check the math:
`left.upper = -3.972967484526552`.
`anchored lower bound = left.upper + gap = -3.972967484526552 + 5.615985475985153e-7 = -3.972966922928004`. Correct.
And the direct right-face enclosure has lower endpoint `-3.972966912717556`.
Wait! `-3.972966912717556` is GREATER than `-3.972966922928004`.
`-3.972966912... > -3.972966922...`
So the direct enclosure IS STRONGER (higher lower endpoint).
Is it mathematically sound to use the direct right-face enclosure if it's stronger?
Yes, a tighter interval bound is always preferable. If you have a verified direct C2 enclosure, you should use it. But the text says "A separate direct right-face C2 enclosure... stronger than the anchored lower bound".
Wait, if they have a direct right-face C2 enclosure, why did they compute the anchored lower bound?
Maybe the anchored lower bound is used for the chart mapping?
"Choose target unstable center as the midpoint of `left.upper` and the anchored right lower bound, and radius as one quarter of their gap. This maps the certified left upper bound to `-2` and right lower bound to `+2`, so both exit faces avoid `[-1,1]` with normalized margin exactly `1`."

Wait. "midpoint of left.upper and the anchored right lower bound".
`anchored right lower bound = left.upper + gap = -3.972966922928005`.
`midpoint = left.upper + gap / 2`.
`radius = gap / 4`.
Wait! If center is midpoint of $A$ and $B$, the radius to cover $A$ to $B$ is $(B-A)/2$.
Here, $A = left.upper = -3.972967484526552$.
$B = anchored right lower bound = -3.972966922928005$.
$B - A = 5.615985475985153e-7$.
So the center is $A + (B-A)/2 = -3.972967484526552 + 2.80799e-7 = -3.972967203727755$.
If radius is $(B-A)/2 = 2.80799e-7$.
Then $A$ maps to Center - Radius = -1.5 ?
Let's re-read: "radius as one quarter of their gap".
Why one quarter of their gap?
If gap is $B-A$, then radius is $(B-A)/4$.
Then $A$ maps to Center - $2 \times Radius = A$.
Wait, if Center is midpoint of $A$ and $B$, then Center $- A = (B-A)/2$.
So if Radius is $(B-A)/4$, then Center $- A = 2 \times Radius$.
Then $A$ maps to $-2$, and $B$ maps to $+2$.
Ah! "This maps the certified left upper bound to `-2` and right lower bound to `+2`".
Is this standard for h-sets?
In a standard h-set covering relation, the target set should be $[-1, 1]^n$.
The coordinates are affine: $x \mapsto \frac{x - \text{center}}{\text{radius}}$.
If $A \mapsto -2$ and $B \mapsto +2$, then the image of $[A, B]$ is $[-2, 2]$.
But the h-set target is $[-1, 1]^2$.
If the left exit face is at $A$ and right exit face is at $B$, and they map to $-2$ and $+2$.
Then the faces avoid $[-1, 1]$ with margin exactly $1$.
Yes, because $[-1, 1]$ is strictly inside $(-2, 2)$.
The stable direction target interval is mentioned next: "The retained Arb support stable image is `[-536870913/1073741824, 536870913/1073741824]`".
$536870913/1073741824 \approx 0.50000000093$.
This is inside $[-1, 1]$.
So the target h-set in the new coordinates is $[-1, 1]^2$.
The image of the left face is at $x \le -2$, right face at $x \ge +2$. This is strictly outside the target h-set.
The image of the stable direction is strictly inside $[-1, 1]$.
This perfectly sets up the covering relation: $N \Rightarrow M$ where $M = [-1, 1]^2$.
The conditions for an h-set covering relation $N \Rightarrow M$ are:
1. $f(\text{exit}(N)) \cap M = \emptyset$
2. $f(\text{entry}(N)) \subset M$ (actually $f(N) \cap M \neq \emptyset$ and $f(\text{entry}) \subset M$).
Here, the stable direction is the entry, and it is strictly inside $[-1, 1]$.
The unstable direction is the exit, and it is strictly outside $[-1, 1]$.
Wait! The standard definition of an h-set covering relation $N \Rightarrow M$ (often called $N \mapsto M$ or $N \overset{f}{\Longrightarrow} M$) requires that the image of the "entry" faces of $N$ covers $M$ in the stable direction, and the "exit" faces avoid $M$.
Specifically, for $N = I_u \times I_s$, $M = J_u \times J_s$:
$f(N) \cap M \neq \emptyset$.
$f(\partial_{\text{entry}} N) \subset \text{Int}(M)$.
$f(\partial_{\text{exit}} N) \cap M = \emptyset$.
Wait! The condition for entry is usually $f(\text{entry}) \subset M$. Here, the stable image is inside $[-1, 1]$, so it is inside $M$.
The condition for exit is $f(\text{exit}) \cap M = \emptyset$. Here, the unstable image is outside $[-2, 2]$, so it avoids $M=[-1,1]$.
Wait, the definition of entry and exit:
In 2D, entry is the stable direction (which contracts), exit is the unstable direction (which expands).
Wait, no! In forward time, the unstable direction EXPANDS, so you EXIT through the unstable faces.
The stable direction CONTRACTS, so you ENTER through the stable faces.
The definition of covering relation $N \Rightarrow M$ means $f(N) \setminus M$ happens only through the exit faces of $M$? No.
Let's check the standard covering relation (e.g., Capinski & Zgliczynski, or Wilczak).
An h-set $N$ has stable direction $s$ and unstable direction $u$.
$N = I_u \times I_s$.
$M = I'_u \times I'_s$.
A covering relation $N \Rightarrow M$ means:
1. $f(N) \cap M \neq \emptyset$.
2. $f(c_u(N) \times \partial I_s) \subset \text{Int}(I'_u \times I'_s)$ ? NO.
Usually, $f(\text{entry}(N)) \subset \text{Int}(M)$, where $\text{entry}(N)$ is the faces in the stable direction.
Wait, no. The "entry" to $M$ is the stable faces of $M$.
So $f(N)$ must enter $M$ through the stable faces of $M$, and leave through the unstable faces of $M$.
Wait, if $M$ is $[-1,1]^2$, then the entry faces of $M$ are the stable ones.
The condition is: $f(N)$ crosses $M$.
In the local coordinate chart of $M$, $M = [-1, 1]_u \times [-1, 1]_s$.
The image of $N$ must have its $u$-coordinate exit $[-1, 1]_u$ on BOTH SIDES.
And its $s$-coordinate must be strictly INSIDE $[-1, 1]_s$.
Let's check the text:
"unstable dimension one, uniform opposite exit signs plus strict positive derivative are used to assign Brouwer degree `+1`."
"The retained Arb support stable image is `[-536870913/1073741824, 536870913/1073741824]`".
This means the stable coordinate of the image is in $[-0.5, 0.5]$ (approx), which is strictly inside $[-1, 1]$.
The unstable coordinate of the image is strictly outside $[-1, 1]$ (it's at $\pm 2$).
Since it has uniform opposite exit signs (left face goes to $-2$, right face goes to $+2$), and the derivative is strictly positive ($>0$ everywhere), the map wraps around the target exactly once.
Thus the Brouwer degree is $+1$.
This satisfies the definition of a covering relation $N \Rightarrow M$.
Wait! The text says: "The target affine chart determinant is positive; the retained physical and rescaled return determinants are strictly negative."
Why does the determinant matter? The return map is the second iterate $P^2$. The map is area-contracting (det < 0). The affine chart determinant is positive, so orientation is preserved in the chart.
Is the degree condition sufficient? "These facts are asserted to certify one local h-set covering relation."
Yes, for a single covering relation $N \Rightarrow M$, verifying the exit/entry inequalities and the Brouwer degree is sufficient to establish the covering relation. You don't need a homotopy/degree condition MISSING. The degree is exactly what is computed.
Wait, to rigorously establish the Brouwer degree, you just need:
- The entry faces (stable direction) map strictly inside the target stable interval.
- The exit faces (unstable direction) map strictly outside the target unstable interval.
- The image crosses the unstable interval.
If the unstable coordinate is strictly monotonic (derivative $> 0$ everywhere) and hits both sides, then the topological degree is 1. This is a standard theorem (e.g., Miranda's theorem or degree theory).
Wait! Is the degree rigorously computed?
Yes, the C2 enclosure showed $\frac{dU}{d\xi} \in [2.86e-7, \dots] > 0$.
So the unstable coordinate is strictly increasing.
Left face maps to $-2$, right face maps to $+2$.
Since it's strictly increasing and covers $[-2, 2]$, it covers $[-1, 1]$ exactly once.
The degree is exactly 1.
So there is no missing homotopy/degree condition. The homotopy to a linear map is implicitly constructed by the monotonicity.
[OK] Do entry avoidance, opposite exit inequalities, chart invertibility, and unstable degree `+1` suffice for this local covering relation? YES.

Let's check the "Chart invertibility and determinants".
"The retained Arb support stable image is `[-536870913/1073741824, 536870913/1073741824]`, giving positive entry margin `536870911/1073741824`."
$536870913/1073741824 = (536870912 + 1) / 1073741824 = 1/2 + 1/2^{30}$.
So the image is $[-1/2 - 2^{-30}, 1/2 + 2^{-30}]$.
The entry margin is $1 - (1/2 + 2^{-30}) = 1/2 - 2^{-30} = 536870912/1073741824 - 1/1073741824 = 536870911/1073741824$.
Math is correct.

Let's check the QR falsifier.
"The dynamic tripleton completed the second section with 2,570 reconditionings. Under the declared L1 max-absolute generator-radius metric, its directional total radius divided by the baseline is approximately `3.51536 > 1`; it therefore fails both improvement and the target `18x` reduction."
This is a statement of fact about a failed computation.
[OK] The math holds (3.5 > 1, so it fails).

Let's look closely at the exact rational gap fraction and the $1/2$ factors.
$U = U(\text{center}) + g_0 d_0 + g_1 d_1 + \frac{1}{2} h_{00} d_0^2 + h_{01} d_0 d_1 + \frac{1}{2} h_{11} d_1^2$.
Wait. The Hessian matrix $H$ has entries:
$H_{00} = 2 (u_x D^2P_{000} + u_y D^2P_{100}) = h_{00}$.
$H_{11} = 2 (u_x D^2P_{001} + u_y D^2P_{101}) = h_{11}$ ? No, the indices for $D^2P$ for second derivative w.r.t $\delta_1$ would be $D^2P_{i02}$? No, the indices in the text are $D^2P_{i00}$ for $\delta_0^2$, $D^2P_{i11}$ for $\delta_1^2$, $D^2P_{i01}$ for $\delta_0 \delta_1$.
Let's verify the indices.
"actual diagonal second derivative is `2*D2P_i00` or `2*D2P_i11`; the off-diagonal actual derivative is `D2P_i01`."
This means the mixed derivative $\frac{\partial^2 P_i}{\partial \delta_0 \partial \delta_1} = D^2P_{i01}$.
The diagonal $\frac{\partial^2 P_i}{\partial \delta_0^2} = 2 D^2P_{i00}$.
So the Hessian matrix $H_i = \begin{pmatrix} 2 D^2P_{i00} & D^2P_{i01} \\ D^2P_{i01} & 2 D^2P_{i11} \end{pmatrix}$.
The Taylor series term is $\frac{1}{2} \delta^T H \delta = \frac{1}{2} ( 2 D^2P_{i00} \delta_0^2 + 2 D^2P_{i01} \delta_0 \delta_1 + 2 D^2P_{i11} \delta_1^2 ) = D^2P_{i00} \delta_0^2 + D^2P_{i01} \delta_0 \delta_1 + D^2P_{i11} \delta_1^2$.
In the text, $U = c \cdot P$, so the second derivative term is:
$c \cdot ( D^2P_{00} \delta_0^2 + D^2P_{01} \delta_0 \delta_1 + D^2P_{11} \delta_1^2 )$.
Wait!
In the text, the Taylor series is written as:
$U = U(\text{center}) + g_0 d_0 + g_1 d_1 + \frac{1}{2} h_{00} d_0^2 + h_{01} d_0 d_1 + \frac{1}{2} h_{11} d_1^2$.
We know $h_{00} = 2(u_x D^2P_{000} + u_y D^2P_{100})$.
So $\frac{1}{2} h_{00} d_0^2 = (u_x D^2P_{000} + u_y D^2P_{100}) d_0^2$.
This perfectly matches $c \cdot D^2P_{00} d_0^2$.
And $h_{01} = u_x D^2P_{001} + u_y D^2P_{101}$.
So $h_{01} d_0 d_1 = (u_x D^2P_{001} + u_y D^2P_{101}) d_0 d_1$.
This perfectly matches $c \cdot D^2P_{01} d_0 d_1$.
So the formula for $U$ is EXACTLY CORRECT.
The notation $h_{00}$ represents the true second derivative, and $h_{01}$ is the true mixed derivative.
Then $\frac{1}{2} h_{00}$ is the Taylor coefficient. This is beautifully consistent.

Let's check the derivative $\frac{dU}{d\delta_0}$.
$\frac{dU}{d\delta_0} = g_0 + h_{00} \delta_0 + h_{01} \delta_1$.
This is the exact derivative of $U$ with respect to $\delta_0$.
[OK] The Taylor enclosure uses correct $1/2$ factors.

Let's check the denominator $154742504910672534362390528000000000000000$.
$154742504910672534362390528000000000000000 = 6.4 \times 24178516392292583494123520000000000000000$.
Wait, $6
