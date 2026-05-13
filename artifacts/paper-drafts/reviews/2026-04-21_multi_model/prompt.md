You are an expert Q1 journal reviewer with joint expertise in clinical EEG,
biomedical biostatistics, and nonassociative algebra. You are being asked for
a rigorous, blunt, pre-submission methodological review of the following
pre-registered preprint.

The preprint is a LaTeX source (reproduced in full below). It reports a
pre-registered cohort study (N=24, CHB-MIT) of a novel scalp-EEG biomarker
based on the sedenion associator norm [a,b,c] = (ab)c - a(bc), with a
Hessian-based pilot retained only as motivation. The PROTOCOL is claimed to
have been registered before analysis.

Primary inferential family under Benjamini-Hochberg FDR at q=0.05:
  T1  pre-ictal dip        p = 0.599       (refuted)
  T2  ictal-onset spike    p = 1.23e-2     (survives, 100k iid perm)
  T4  LOO logistic AUC     0.642 [0.513, 0.764], p = 1.63e-2 (survives, 20k boot)

Supporting, not in the BH-FDR family:
  T2 also significant under circular-shift null (p=8.99e-5) and
     10-epoch block bootstrap (p=3.07e-2)
  T3  channel-subset robustness: 100/100 sign preservation on dip and
     spike across 2400 runs (24 patients × 100 sixteen-channel draws of
     23 canonical channels)
  T5  co-occurrence of dip+spike, p = 0.31, reported for completeness

Please produce a review with the following five sections, each in plain
prose (no LaTeX). Be concrete. Cite the preprint's own wording when you
raise a concern. Do not be polite.

  1. "Top 5 concerns, ranked by threat to the main claim."
     Focus on whether the paper actually supports the central claim that
     "the sedenion-associator norm spikes at ictal onset", as opposed to
     showing a milder statistical effect over a small cohort.

  2. "Statistical / pre-registration audit."
     Check the BH-FDR family definition. Check whether the three nulls
     (iid, circular, block) are treated as independent evidence or as
     one decision, and whether the choice of iid as primary is
     pre-specified. Check the LOO AUC 95% CI width against N=24 and
     flag whether it is reported honestly. Check whether the pilot
     Hessian study could have leaked the choice of statistic.

  3. "Biomarker construction and algebraic choices."
     Is the sedenion associator norm well-motivated or arbitrary? Would
     a simpler nonlinearity (cubic coupling, triple product of whitened
     channels, tensor associator of quaternions) produce the same
     signal? Is the sedenion algebra load-bearing or ornamental?

  4. "Clinical validity and scope."
     Given scalp EEG, 24 pediatric patients, CHB-MIT (single center,
     enriched), can the paper honestly call this "evidence" of a
     generic pre-ictal/ictal mechanism? What is the right caveat?

  5. "Recommendation: accept / major revisions / reject, with the single
     most important change required before submission."

At the end, list any passages that would, in your judgment, trigger a
methodological-review desk rejection if left as-is.

---- BEGIN PREPRINT TEX ----
\documentclass[11pt,twocolumn]{article}
\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage{amsmath,amssymb,amsfonts}
\usepackage{booktabs}
\usepackage{graphicx}
\usepackage{hyperref}
\usepackage{xcolor}
\usepackage[margin=2cm]{geometry}
\usepackage{authblk}
\usepackage{multirow}

\title{Pre-Registered Evidence that the Sedenion Associator Norm\\
Spikes at Ictal Onset in Scalp EEG}

\author[1,2]{Demetrios Chiuratto Agourakis\thanks{ORCID: \href{https://orcid.org/0009-0001-8671-8878}{0009-0001-8671-8878}. Corresponding author: \texttt{demetrios@agourakis.med.br}.}}
\author[1]{Marli Gerenutti\thanks{ORCID: \href{https://orcid.org/0000-0001-7165-646X}{0000-0001-7165-646X}}}
\affil[1]{Pontif\'icia Universidade Cat\'olica de S\~ao Paulo (PUC-SP), Sorocaba, Brazil}
\affil[2]{Faculdade S\~ao Leopoldo Mandic --- Campinas, Brazil}

\date{April 2026}

\begin{document}
\maketitle

\begin{abstract}
Non-associativity is a geometric property: it measures the failure of
$(a\otimes b)\otimes c$ to equal $a\otimes(b\otimes c)$. In a hypercomplex
state-space model (SSM) of EEG, the ternary object that carries this
information is the \emph{associator}
$[a,b,c]=(a\otimes b)\otimes c-a\otimes(b\otimes c)$, and its norm
$\|[a,b,c]\|_2$ is a scalar summary of how order-sensitive the current
recurrence is. We report a pre-registered test of whether the sedenion
associator norm changes at seizure onset in the CHB-MIT Scalp EEG
Database ($N=24$ patients, 16 channels, 256\,Hz). The pre-registration
fixes the cohort, the six analysis windows ($\mathrm{FAR},\,\mathrm{PRE30},\,
\mathrm{PRE10},\,\mathrm{PRE5},\,\mathrm{IC},\,\mathrm{POST}$), the
per-patient statistics, the family of tests (T1--T5), the null models,
and the correction rule (Benjamini--Hochberg at $q=0.05$ on the primary
family \{T1, T2, T4\}). Results after 100{,}000 within-patient
permutations and 20{,}000 bootstrap iterations: the ictal-onset spike
(T2) survives BH-FDR with Fisher-combined $p=1.23\times10^{-2}$ under
iid nulls, $8.99\times10^{-5}$ under circular shifts, and
$3.07\times10^{-2}$ under 10-epoch block bootstrap; the pre-ictal dip
(T1) is refuted ($p=0.599$); a leave-one-patient-out logistic
classifier on $(\mathrm{dip},\mathrm{spike})$ reaches
$\mathrm{AUC}=0.642$ with 95\,\% CI $[0.513,0.764]$ and one-sided
bootstrap $p=1.63\times10^{-2}$ (T4); and a 100-draw channel-subset
robustness sweep (T3, 2400 runs) preserves the sign of both the cohort
dip and the cohort spike in 100/100 draws. The ictal-onset spike in the
associator norm is therefore a pre-registered, sign-consistent, and
spatially robust finding; it is not an artefact of a particular channel
montage, permutation scheme, or classifier choice. We release the full
pipeline in the Sounio language and commit every intermediate artefact
to the repository.
\end{abstract}

\section{Introduction}

Focal seizures are geometric events. During ictal onset the cortical
state traces a trajectory whose order matters: the same three neural
events delivered in different sequences produce different downstream
dynamics \cite{lopes2003model,jirsa2014nature}. A state-space model
whose recurrence operator lives in an \emph{associative} algebra
(real, complex, quaternion) cannot express that order dependence by
construction --- $(a\cdot b)\cdot c=a\cdot(b\cdot c)$ no matter what the
signal does. A recurrence in a \emph{non-associative} algebra can, and
the amount by which it does is measurable.

The associator
\begin{equation}
  [a,b,c] \;=\; (a\otimes b)\otimes c \;-\; a\otimes(b\otimes c)
  \label{eq:associator}
\end{equation}
is the canonical algebraic object that records the failure of
associativity \cite{baez2002octonions,schafer1966}. It is identically
zero in any associative algebra, non-zero on generic octonion triples,
and grows in magnitude once zero divisors are available at and above
dimension 16 (sedenions) \cite{moreno1998zero,kinyon2004zero}. Its norm
$\|[a,b,c]\|_2$ is a scalar that can be attached to any epoch of an
EEG recording once a hypercomplex lifting is fixed.

This paper asks a single empirical question:
\begin{quote}
\emph{Does the sedenion associator norm computed from a 16-channel
scalp EEG window change at seizure onset in a cohort of patients, in a
way that cannot be explained by chance, by the specific channel
subset, or by post-hoc selection?}
\end{quote}

We answer this with a pre-registered protocol that was committed to
the repository before the inferential tests were run
(PROTOCOL.md, git SHA in \S\ref{sec:reproducibility}). The protocol
names:
\begin{itemize}
\item five hypotheses and their one-sided directions (H1--H5);
\item five tests (T1--T5) mapped onto those hypotheses;
\item three null models for the permutation arm (iid shuffle,
      circular shift, 10-epoch block bootstrap);
\item a primary family \{T1, T2, T4\} under
      Benjamini--Hochberg at $q=0.05$, with T3 and T5 declared
      supporting; and
\item the exact window definitions, channel map, permutation
      resolution, and random seeds.
\end{itemize}

Our headline result is that the ictal-onset spike (T2) and the
LOO classifier (T4) both survive joint BH-FDR control, the pre-ictal
dip (T1) is refuted, and the channel-subset robustness test (T3) is
saturated at 100\,\% sign preservation on both arms. Because the
protocol was frozen in advance and the nulls are exchangeable within
patient, the residual $p$ of $1.2\times10^{-2}$ on T2 is a calibrated
false-positive budget, not a selection-biased estimate.

Two paragraphs of framing are important before the methods.

\paragraph{Relationship to a prior Hessian pilot.}
An earlier analysis proposed the off-diagonal $L_1$ norm of the
Hessian of the training loss as a candidate biomarker of
non-associative curvature \cite{agourakis2026hessian_pilot}. In our
hands that diagnostic was unstable: it required a fitted model, it
depended sensitively on optimiser hyperparameters, and on a small
pilot ($n\le7$) the direction of its ictal/baseline contrast flipped
across patients. We therefore abandoned it in favour of the associator
norm, which is a \emph{data-level} statistic: it is computed directly
from a short signal window via a fixed hypercomplex lifting and
requires no training. The price is that the associator norm indexes
the geometry of the signal and its sedenion lifting, not the geometry
of a learned model. The Hessian pilot is summarised here only to
explain why we did not report its numbers as a primary result.

\paragraph{Why sedenions and not octonions.}
Octonions ($\dim 8$) are non-associative but \emph{alternative}: every
pair generates an associative subalgebra, so the associator vanishes
as soon as two of the three arguments coincide in a quaternionic
subalgebra. Sedenions ($\dim 16$) are non-associative and
non-alternative, and they contain zero divisors
\cite{moreno1998zero,imaeda2000}. For a 16-channel EEG window the
sedenion lifting is the minimal Cayley--Dickson algebra whose
associator has no alternative-law cancellation, so its norm is a more
faithful index of order-sensitivity than the octonion associator
would be. This choice is pre-registered and not tuned to the data.

\section{Background}

\subsection{Cayley--Dickson algebras and the associator}
The Cayley--Dickson construction generates, from $\mathbb{R}$, a
sequence of power-associative real algebras of dimension $2^k$:
$\mathbb{R}\to\mathbb{C}\to\mathbb{H}\to\mathbb{O}\to\mathbb{S}\to\ldots$
\cite{schafer1966}. Associativity is lost at $k=3$ (octonions,
$\mathbb{O}$); the alternative law is lost at $k=4$ (sedenions,
$\mathbb{S}$); zero divisors appear at $k=4$
\cite{moreno1998zero,kinyon2004zero}. We work exclusively with the
sedenion algebra $\mathbb{S}$, identified with $\mathbb{R}^{16}$ and
equipped with the standard Cayley--Dickson product.

The \emph{associator} of three sedenions is defined by
equation~(\ref{eq:associator}). It is trilinear, alternating on any
two inputs that lie in the same quaternionic subalgebra, and
vanishing whenever the three inputs are jointly octonionic. We use
$\|\cdot\|_2$ throughout; the choice of Euclidean norm is arbitrary
but consistent.

\subsection{Sedenion lifting of a 16-channel EEG window}
Let $s\in\mathbb{R}^{16\times T}$ be a 16-channel EEG window of
length $T=80$ samples (at 256\,Hz, $T\approx 312$\,ms). We fix a
channel map $\chi:\{1,\ldots,16\}\to\mathbb{S}$ that sends channel
$i$ to the standard basis element $e_{i-1}$, and embed the window as
three sedenion signals:
\begin{align}
  a_t &= \textstyle\sum_{i=1}^{16} \bar{s}_{i,t}\,e_{i-1}, \\
  h_t &= \textstyle\sum_{i=1}^{16} \bar{s}_{i,t-1}\,e_{i-1}, \\
  x_t &= \textstyle\sum_{i=1}^{16} \bar{s}_{i,t-2}\,e_{i-1},
\end{align}
with $\bar{s}$ the within-training normalisation described in
\S\ref{sec:methods}. The associator trajectory
$t\mapsto \|[a_t,h_t,x_t]\|_2$ is a real scalar sequence; we reduce
it to a single number per window by taking the mean over the 80
time-steps. We refer to this scalar as ``the associator norm'' of the
window.

\subsection{Pre-ictal dip and ictal spike}
Fix a seizure onset time $t^{\ast}$ and consider six windows:
$\mathrm{FAR}$ at $t^{\ast}-60\text{s}$,
$\mathrm{PRE30}$ at $t^{\ast}-30\text{s}$,
$\mathrm{PRE10}$ at $t^{\ast}-10\text{s}$,
$\mathrm{PRE5}$ at $t^{\ast}-5\text{s}$,
$\mathrm{IC}$ at $t^{\ast}$, and
$\mathrm{POST}$ at $t^{\ast}+5\text{s}$.
Let $A_w$ denote the associator norm of window $w$.
The two pre-registered per-patient statistics are
\begin{align}
  \mathrm{dip} &=
    \frac{A_{\mathrm{FAR}}
    - \min(A_{\mathrm{PRE30}},A_{\mathrm{PRE10}},A_{\mathrm{PRE5}})}
    {A_{\mathrm{FAR}}},
    \label{eq:dip}\\
  \mathrm{spike} &=
    \frac{A_{\mathrm{IC}} - A_{\mathrm{PRE5}}}{A_{\mathrm{PRE5}}}.
    \label{eq:spike}
\end{align}
Equation~(\ref{eq:dip}) is positive iff the minimum pre-ictal
associator norm is below the far baseline; (\ref{eq:spike}) is
positive iff the ictal-onset window exceeds the immediate
pre-ictal window. Both statistics are expressed as fractional
deviations and are therefore dimensionless.

\section{Methods}\label{sec:methods}

\subsection{Data}
We use the public CHB-MIT Scalp EEG Database
\cite{shoeb2009} as distributed on PhysioNet
\cite{goldberger2000physionet}. Each patient contributes one ictal
recording, segmented to $\pm 90$\,s around the first annotated
seizure onset. Sampling rate is $256$\,Hz; $N=24$ patients are
retained (the full public cohort; the manifest is committed as
\texttt{scripts/research/door\_f\_cohort/chbmit\_manifest.tsv}).
For the primary analysis, channels 1--16 are used in EDF order
($\mathrm{CH\_MAP}=\mathrm{range}(16)$) for every patient; the
T3 robustness arm draws 100 random 16-of-23 subsets per patient
(\S\ref{sec:t3}).

\subsection{Pre-processing and normalisation}
For each patient we form a 7-window decomposition at
$\{\mathrm{II},\mathrm{FAR},\mathrm{PRE30},\mathrm{PRE10},
\mathrm{PRE5},\mathrm{IC},\mathrm{POST}\}$, each of length
$T=80$ samples. The interictal-training window $\mathrm{II}$ is
placed at $t^{\ast}-300\text{s}$ and used only to estimate the
per-channel mean, standard deviation, and maximum absolute value
from its first 64 samples; the remaining six probe windows are
standardised with these statistics and clipped to $\pm 5$ before
sedenion lifting. The normalisation pipeline is identical to the
production generator committed under
\texttt{scripts/research/door\_f\_cohort/generate.py}.

\subsection{Associator-norm pipeline}
For each probe window the following is executed inside a single
Sounio program compiled to native x86-64 through the self-hosted
compiler (no Python numerics in the hot path):
\begin{enumerate}
\item Lift each of the three time-shifted 16-channel samples to
      $\mathbb{S}$ using the fixed channel map.
\item Compute the associator $[a_t,h_t,x_t]$ using the
      Cayley--Dickson product.
\item Average $\|[a_t,h_t,x_t]\|_2$ over the 80 time-steps of the
      window and emit the scalar.
\end{enumerate}
The same program emits the mean squared prediction error of an
autoregressive baseline for comparison; MSE is not part of the
primary analysis but is retained in the per-patient TSV.

\subsection{Hypotheses and tests}
The full pre-registered protocol is
\texttt{artifacts/research/door\_f\_cohort\_N24/PROTOCOL.md} and
states:
\begin{itemize}
\item \textbf{H1 (dip).} The minimum pre-ictal associator norm is
      strictly below the far baseline, one-sided.
\item \textbf{H2 (spike).} The ictal-onset window exceeds the
      immediate pre-ictal window, one-sided.
\item \textbf{H3 (co-occurrence).} Dip and spike co-occur at a rate
      above the independence product, one-sided.
\item \textbf{H4 (classification).} A two-feature classifier on
      $(\mathrm{dip},\mathrm{spike})$ separates ictal from non-ictal
      windows at leave-one-patient-out
      $\mathrm{AUC}>0.5$ with the lower 95\,\% bootstrap bound
      strictly above $0.5$.
\item \textbf{H5 (channel robustness).} Both dip and spike signs
      are preserved in at least 95\,\% of random 16-channel
      subsets.
\end{itemize}
The mapping to tests is given in Table~\ref{tab:tests}.

\begin{table}[t]
\centering\small
\begin{tabular}{@{}llp{3.4cm}@{}}
\toprule
Test & Hypothesis & Statistic \\
\midrule
T1 & H1 & Fisher-combined $p$, 100k within-patient permutations \\
T2 & H2 & same, spike statistic \\
T3 & H5 & 100 draws $\times$ 16-of-$\ge 23$ channels per patient,
            2400 runs total \\
T4 & H4 & LOO logistic on $(\mathrm{dip},\mathrm{spike})$,
            DeLong-equivalent bootstrap CI \\
T5 & H3 & observed $\#(\mathrm{dip}>0,\mathrm{spike}>0)$
            vs independence, $10^4$ Monte-Carlo draws \\
\bottomrule
\end{tabular}
\caption{Pre-registered test family. Primary family
         \{T1, T2, T4\} carries a BH correction at $q=0.05$;
         T3 and T5 are declared supporting.}
\label{tab:tests}
\end{table}

\subsection{Null models}
We report every permutation-arm $p$ under three exchangeable nulls:
\textbf{iid} (uniform permutation of epoch labels within patient),
\textbf{circular shift} (random rotation of the per-epoch label
sequence, which preserves temporal autocorrelation), and
\textbf{10-epoch block bootstrap} (resampling of non-overlapping
10-epoch blocks of labels). All three are within-patient; no
cross-patient permutation is performed. The iid null is the primary
per-pre-registration; the other two are reported as robustness
checks and are commented on in \S\ref{sec:results_nulls}.

\subsection{Stopping rule, seeds, and corrections}
No stopping rule applies: every patient in the public CHB-MIT
cohort with an annotated ictal onset is retained. Random seeds are
frozen in \texttt{PROTOCOL.md} (seed \texttt{20260420} for T4/T5,
\texttt{20260421} for T3). The primary family \{T1, T2, T4\}
is corrected under Benjamini--Hochberg at $q=0.05$
\cite{benjamini1995bh}; the supporting family \{T3, T5\} is
reported without correction.

\section{Results}\label{sec:results}

\subsection{Cohort-level window trajectory}
Figure~\ref{fig:window} shows the associator norm across the six
probe windows for all 24 patients and the cohort median. At ictal
onset the cohort median rises from $1.72$ at $\mathrm{PRE5}$ to
$2.11$ at $\mathrm{IC}$ (a $+22.7\,\%$ spike on the median), and
continues to $2.78$ at $\mathrm{POST}$. The pre-ictal windows
$\{\mathrm{PRE30},\mathrm{PRE10},\mathrm{PRE5}\}$ show no consistent
dip relative to $\mathrm{FAR}$: the cohort median of the dip
statistic is $+0.17$, driven by a minority of patients.

\begin{figure}[t]
\centering
\includegraphics[width=\linewidth]{figures/fig_window_trajectory.pdf}
\caption{Associator norm $\|[a,h,x]\|_2$ across the six analysis
         windows for all $N=24$ patients (grey) and the cohort
         median (blue). Vertical axis is logarithmic. The ictal-onset
         spike from $\mathrm{PRE5}$ to $\mathrm{IC}$ is visible in
         the cohort trajectory; the pre-ictal dip is not.}
\label{fig:window}
\end{figure}

\subsection{Pre-registered primary family}
Table~\ref{tab:primary} reports the joint Benjamini--Hochberg
control on \{T1, T2, T4\}. T2 (ictal spike) and T4 (LOO
classifier) both survive at $q=0.05$; T1 (pre-ictal dip) is
refuted.

\begin{table}[t]
\centering\small
\begin{tabular}{@{}llrrc@{}}
\toprule
Rank & Test & $p$ & BH cut & Survives? \\
\midrule
1 & T2 ictal spike       & $1.23\times10^{-2}$ & $0.0167$ & \checkmark \\
2 & T4 LOO AUC $=0.642$  & $1.63\times10^{-2}$ & $0.0333$ & \checkmark \\
3 & T1 pre-ictal dip     & $0.599$             & $0.05$   & $\times$ \\
\bottomrule
\end{tabular}
\caption{Primary family under BH-FDR at $q=0.05$. $p$-values are
         $10^5$ within-patient iid permutations (T1, T2) and
         $2\times10^4$ stratified bootstrap resamples (T4).}
\label{tab:primary}
\end{table}

\subsection{T2 ictal spike is robust across three null models}
\label{sec:results_nulls}
Figure~\ref{fig:perm} reports the Fisher-combined $p$-value of the
spike statistic under each of the three null models. The effect is
smallest under iid ($1.23\times10^{-2}$) and strongest under
circular shift ($8.99\times10^{-5}$); the 10-epoch block bootstrap,
which preserves more temporal structure, gives $p=3.07\times10^{-2}$.
All three are below $\alpha=0.05$; the dip statistic does not survive
any of the three.

\begin{figure}[t]
\centering
\includegraphics[width=\linewidth]{figures/fig_permutation_summary.pdf}
\caption{Fisher-combined $p$-value across three exchangeable null
         models for T2 (spike) and T1 (dip). Dashed line at
         $\alpha=0.05$. T2 survives all three nulls; T1 survives
         none.}
\label{fig:perm}
\end{figure}

\subsection{T4 leave-one-patient-out classification}
Training a logistic regression on z-scored
$(\mathrm{dip},\mathrm{spike})$ features across $N-1$ patients
and scoring the seven windows of the held-out patient
yields a pooled $\mathrm{AUC}=0.642$ with 95\,\% bootstrap CI
$[0.513,0.764]$ and one-sided bootstrap $p=1.63\times10^{-2}$
(Figure~\ref{fig:roc}). The lower bound of the CI is strictly above
chance, satisfying the pre-registered decision rule for H4.

\begin{figure}[t]
\centering
\includegraphics[width=0.85\linewidth]{figures/fig_loo_roc.pdf}
\caption{T4: pooled ROC of the leave-one-patient-out logistic
         classifier on $(\mathrm{dip},\mathrm{spike})$. Label is
         $\mathrm{IC}=1$ vs the six non-ictal windows.}
\label{fig:roc}
\end{figure}

\subsection{T3 channel-subset robustness}\label{sec:t3}
For every patient we draw $100$ random 16-of-$\ge 23$ channel
subsets using \texttt{numpy.random.SeedSequence(20260421)} with
per-patient spawn keys, regenerate the full window decomposition
for each draw, and recompute the per-patient dip and spike. The
full sweep is 2400 (compile + run) executions. The decision rule
declares T3 robust iff at least 95 of the 100 cohort medians
preserve the canonical positive sign, \emph{separately} for dip
and for spike. Figure~\ref{fig:t3} shows the result: sign
preservation is saturated at $100/100$ for both axes. The
cohort-median dip across draws is $0.328$ with 95\,\% percentile
interval $[0.257,0.375]$; the cohort-median spike is $0.227$ with
95\,\% interval $[0.124,0.345]$.

\begin{figure}[t]
\centering
\includegraphics[width=0.85\linewidth]{figures/fig_t3_robust.pdf}
\caption{T3: joint distribution of the cohort-median dip and
         spike across 100 random 16-of-23 channel subsets. Every
         draw falls in the upper-right quadrant (sign preserved).}
\label{fig:t3}
\end{figure}

\subsection{T5 co-occurrence (supporting)}
The observed number of patients with simultaneously positive
dip and spike is $12/24$, compared to an independence-product
expectation of $10.5$; one-sided Monte-Carlo $p = 0.31$. T5 does
not add evidence beyond T2 and is reported only for completeness.

\section{Discussion}

The sedenion associator norm spikes at ictal onset in the CHB-MIT
cohort, pre-registered, 100k-permutation, BH-FDR-controlled. The
effect is small on the median ($+22.7\,\%$ from PRE5 to IC) but
consistent: it survives three different null models, it is
spatially robust at 100/100 channel subsets, and it is recoverable
by a two-feature LOO classifier with $\mathrm{AUC}=0.642$.

\paragraph{What the associator norm is measuring.}
Equation (\ref{eq:associator}) is sensitive to the degree to
which the three sedenion embeddings fail to lie jointly in a
common octonionic subalgebra. In a quasi-stationary baseline
window the three time-shifted samples $a_t, h_t, x_t$ are highly
collinear (autocorrelation is high at 256\,Hz over a 2-sample lag),
so the three embeddings are close in $\mathbb{S}$ and the
associator is small. At ictal onset the signals become less
autoregressive and more phase-diverse across channels
\cite{jirsa2014nature,schindler2007}: the three embeddings move
apart in the Cayley--Dickson sense, and the associator norm grows.
The Euclidean norm of $[a,h,x]$ is therefore an empirical,
data-level index of how far the recent recurrence structure
departs from an octonion subalgebra --- it is a geometry observable
of the signal, not of any trained model.

\paragraph{Why the pre-ictal dip fails.}
The dip statistic (\ref{eq:dip}) presumes a brief
quasi-harmonic state that reduces order-sensitivity before
the transition. The literature for scalp EEG on this kind of
prodromal signature is mixed \cite{mormann2007seizure,litt2001}:
our cohort shows no cohort-level dip once the 100k-permutation
null is enforced. We register this as a refuted prediction and do
not reinterpret it post hoc.

\paragraph{Relationship to prior work.}
Non-associative operators in SSMs have been proposed on purely
architectural grounds \cite{gu2022efficiently} but have not, to
our knowledge, been evaluated as a signal biomarker. The
octonion-associator work in systems biology
\cite{agourakis2026fano} and the sedenion zero-divisor geometry
in \cite{cawagas2004,kinyon2004zero} motivate the choice of
$\mathbb{S}$ over $\mathbb{O}$: sedenions are the smallest
Cayley--Dickson algebra in which the alternative law fails, so
their associator is the minimal probe of true non-associative
structure.

\paragraph{Limitations.}
Three limitations are immediate:
(i) CHB-MIT contains pediatric intractable epilepsy
    \cite{shoeb2009}; we cannot claim the associator spike as a
    general onset marker outside this population.
(ii) The within-patient iid null has $p=1.23\times10^{-2}$, which
     is modest; the confidence interval of the T4 LOO AUC is wide
     ($[0.513,0.764]$) and the lower bound barely clears chance.
(iii) We do not claim a causal physiological mechanism. The
      associator norm tracks a property of the sedenion lifting,
      not a named neural variable.

\section{Reproducibility}\label{sec:reproducibility}
All primary artefacts live under
\texttt{artifacts/research/door\_f\_cohort\_N24/} and every
number in this paper can be regenerated from:
\begin{itemize}
\item \texttt{PROTOCOL.md} --- pre-registration, frozen before
      any inferential test was run;
\item \texttt{door\_f\_cohort.tsv} --- per-patient per-window
      associator and MSE values;
\item \texttt{p5\_permutation\_100k/} ---
      \texttt{cohort.tsv}, \texttt{per\_patient.tsv},
      \texttt{bh\_fdr.tsv}, \texttt{t4\_loo.json},
      \texttt{t4\_loo\_predictions.tsv};
\item \texttt{p5\_t3\_channel\_robust/} ---
      \texttt{draw\_manifest.tsv},
      \texttt{t3\_per\_draw.tsv},
      \texttt{t3\_cohort\_by\_draw.tsv},
      \texttt{t3\_robust.json}.
\end{itemize}
The driver scripts are committed under
\texttt{scripts/research/door\_f\_cohort/}; the figure generator
is \texttt{scripts/research/door\_f\_cohort/make\_preprint\_figures.py}
and reads only from the artefact tree above. The self-hosted
compiler binary used for every run is \texttt{bin/souc} at the
SHA named by the commit accompanying this draft.

\section{Conclusion}
A pre-registered, 100k-permutation, BH-FDR-controlled analysis
on the full 24-patient public CHB-MIT cohort finds that the
sedenion associator norm computed from a 16-channel EEG window
spikes at seizure onset. The effect is significant under iid
and circular-shift nulls, survives a 10-epoch block bootstrap,
is recoverable by a two-feature LOO classifier with a CI lower
bound strictly above chance, and is spatially robust at
$100/100$ random 16-channel subsets. The pre-ictal dip
hypothesis is refuted. The associator norm is therefore a
candidate, data-level, non-associative-geometry observable of
ictal onset; its extension to broader populations, its
comparison against established univariate EEG markers, and its
integration into a detection pipeline are left for future work.

\section*{Acknowledgments}
We thank the MIT CHB-MIT contributors \cite{shoeb2009} and the
PhysioNet project \cite{goldberger2000physionet} for the public
availability of the EEG data, without which a pre-registered
multi-patient test of a new biomarker would not be feasible in
an academic setting.

\begin{thebibliography}{99}
\bibitem{baez2002octonions} J. C. Baez.
 The octonions.
 \emph{Bull. Amer. Math. Soc.}, 39(2):145--205, 2002.
\bibitem{schafer1966} R. D. Schafer.
 \emph{An Introduction to Nonassociative Algebras}.
 Academic Press, 1966.
\bibitem{moreno1998zero} G. Moreno.
 The zero divisors of the Cayley--Dickson algebras over the real
 numbers.
 \emph{Bol. Soc. Mat. Mexicana (3)}, 4:13--28, 1998.
\bibitem{kinyon2004zero} M. K. Kinyon and J. D. Phillips.
 Axioms for trimedial quasigroups.
 \emph{Comment. Math. Univ. Carolin.}, 45:287--294, 2004.
\bibitem{imaeda2000} K. Imaeda and M. Imaeda.
 Sedenions: algebra and analysis.
 \emph{Appl. Math. Comput.}, 115(2--3):77--88, 2000.
\bibitem{cawagas2004} R. E. Cawagas.
 On the structure and zero divisors of the Cayley--Dickson
 sedenion algebra.
 \emph{Discuss. Math. Gen. Algebra Appl.}, 24:251--265, 2004.
\bibitem{shoeb2009} A. H. Shoeb.
 \emph{Application of Machine Learning to Epileptic Seizure
 Onset Detection and Treatment}.
 Ph.D. thesis, MIT, 2009.
\bibitem{goldberger2000physionet} A. L. Goldberger et al.
 PhysioBank, PhysioToolkit, and PhysioNet.
 \emph{Circulation}, 101(23):e215--e220, 2000.
\bibitem{lopes2003model} F. H. Lopes da Silva et al.
 Epilepsies as dynamical diseases of brain systems.
 \emph{Epilepsia}, 44(s12):72--83, 2003.
\bibitem{jirsa2014nature} V. K. Jirsa et al.
 On the nature of seizure dynamics.
 \emph{Brain}, 137(8):2210--2230, 2014.
\bibitem{schindler2007} K. Schindler et al.
 Assessing seizure dynamics by analysing the correlation
 structure of multichannel intracranial EEG.
 \emph{Brain}, 130(1):65--77, 2007.
\bibitem{mormann2007seizure} F. Mormann et al.
 Seizure prediction: the long and winding road.
 \emph{Brain}, 130(2):314--333, 2007.
\bibitem{litt2001} B. Litt and J. Echauz.
 Prediction of epileptic seizures.
 \emph{Lancet Neurol.}, 1(1):22--30, 2002.
\bibitem{gu2022efficiently} A. Gu, K. Goel, and C. R\'e.
 Efficiently modeling long sequences with structured state
 spaces.
 In \emph{ICLR}, 2022.
\bibitem{benjamini1995bh} Y. Benjamini and Y. Hochberg.
 Controlling the false discovery rate: a practical and
 powerful approach to multiple testing.
 \emph{J. R. Stat. Soc. B}, 57(1):289--300, 1995.
\bibitem{agourakis2026hessian_pilot}
 D. C. Agourakis and M. Gerenutti.
 Non-associative curvature of octonion state-space models:
 a Hessian pilot (unstable, superseded).
 Internal report, Sounio project, March 2026.
\bibitem{agourakis2026fano}
 D. C. Agourakis and M. Gerenutti.
 $\mathrm{PG}(k-1,2)$ geometry as a mathematical framework for
 order-dependent processes in biology.
 Preprint, March 2026.
\end{thebibliography}

\appendix

\section{Per-patient primary statistics}\label{app:perpatient}
Table~\ref{tab:perpatient} lists, for each of the 24 patients,
the observed dip and spike and their per-patient $p$-values
under the iid null. All patients are retained regardless of
the observed statistics; no patient is excluded post hoc.

\begin{table}[h]
\centering\small
\begin{tabular}{@{}lrrrr@{}}
\toprule
Patient & dip & spike & $p_{\mathrm{dip}}$ & $p_{\mathrm{spike}}$ \\
\midrule
\input{tables/perpatient_iid.tex}
\bottomrule
\end{tabular}
\caption{Per-patient dip and spike (observed) and per-patient
one-sided $p$-values under the iid permutation null
($10^5$ draws). Source:
\texttt{p5\_permutation\_100k/per\_patient.tsv}.}
\label{tab:perpatient}
\end{table}

\end{document}
---- END PREPRINT TEX ----
