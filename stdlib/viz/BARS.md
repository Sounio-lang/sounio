# Checked bars over the shared scientific Visual IR

## API and geometry

`viz::bar::bar_pair_width(x0,x1,pair_count,xmin,xmax,x,w)->BarWidth`
returns value/status. Count must be0..2. Geometry/domain are always checked;
count0 ignores both samples, count1 validates x0 and ignores x1, count2 validates
and projects both. Counts0/1 return width8. Count2 uses the absolute gap between
the two projected integer centers times8/10, clamped to2..40. Duplicate centers
therefore produce width2. No sorting or spacing inference is performed.

`bar_project(sx,sy,xmin,xmax,ymin,ymax,x,y,w,h,width,clipped)->BarRect`
returns x/y/w/h/status. Width must be2..40. Status0 has positive dimensions;
1 means empty or outside;2 means invalid. All unavailable fields are zero.
The projector allocates no heap and has no IR dependency.

Finite sample/domain/layout checks and conservative normalized-coordinate limits
reuse scatter projection before floating-to-integer casts. Layout endpoints must
be within±1,000,000, extents1..1,000,000; nonconstant spans must be finite positive.
Constant x accepts only an exactly matching sample and maps integer width/2.
Constant y is accepted only when both bound and sample are zero: the bar is empty.
Other constant y domains are invalid. These are explicit finite-range contracts,
not arbitrary binary64 projection guarantees.

Nonconstant centers, tips and zero baseline retain truncation from
`scatter_project`. In legacy mode the baseline is actual data zero, even when
outside a caller-supplied nonconstant y domain. Auto-domain construction includes
zero, so ordinary built bars do not accidentally crop that baseline.

## Clipped bars and windows excluding zero

`viz_bar_set_viewport(scene,node,xmin,xmax,ymin,ymax)->bool` validates a bar node,
slot, untransformed flags, bounded layout, opaque RGB8/stroke1 and strictly
increasing finite domains/spans before any mutation. It sets shared flag256;
domain storage belongs to the series and is visible to any node sharing it.

Before projecting a clipped bar, its tip is clamped vertically to the y window
and its zero baseline is clamped separately. Thus a positive window[5,10] with
a value8 preserves the segment[5,8]; it does not redefine the scientific zero.
A bar entirely below/above a one-sided window may be empty. This is geometric
cropping, not a changed baseline convention.

Horizontal centers are NOT culled against the raw x domain. A center just outside
can have a footprint intersecting the plot and remains partly visible.
Actual centers are projected, then rectangles are formed with
left=center-width/2, right=left+width, top=min(tip,baseline),
bottom=max(tip,baseline). Those half-open bounds are intersected with
[x,x+w+1)×[y,y+h+1), corresponding to closed pixel-center plot bounds.
Empty intersections return1. An excessively distant/unprojectable center returns2,
not a fabricated boundary position.

## Builder and render routing

Only `viz_add_bar_chart` changes auto-domain policy: bounds use paired finite
rows, y bounds include zero, raw arrays and n are retained. Nonempty all-invalid
input returns-1 transactionally; empty input remains allowed. Nonfinite rows are
not converted to integer coordinates. Other chart/series builders are unchanged.

Both renderers scan input order once to find the first two paired finite rows
BEFORE crop and use their x values for shared width. Width therefore does not
depend on which rows survive the viewport. This is the documented legacy spacing
policy, not an estimate suitable for all irregular/histogram bins.

Canvas fills exactly the shared half-open integer rectangle. SVG emits its pixel
cells as x=rect.x-.5,y=rect.y-.5,width=rect.w,height=rect.h. No additional clipPath
is required because the rectangle is already intersected. SVG/Canvas
antialiasing is not promised identical. Clipped Canvas bars draw no implicit axes;
legacy axes/frame retain zero-line semantics through checked scatter projection.

SVG retains `<!-- bar n=N -->` and adds
`<!-- bar counts visible=V empty_or_outside=O invalid=I -->`.
Counts describe input rows, not unique centers or nonoverlapping rectangles.

## Scientific and resource limits

No stacked/grouped bars, explicit bin edges, weighted histogram estimator,
log/symlog bar transform, uncertainty interval, statistical inference or automatic
axis dependency is introduced. Raw viewport setters reject transformed flags.
Negative/zero values are not replaced by epsilon. A scientific consumer must label
a vertically cropped window and any transformation applied to observations.

Shared helpers use scalar storage; builders add a finite-pair scan before existing
series copy and renderers add a bounded first-pair scan. No measured runtime,
RSS, artifact-size superiority or universal renderer validation is claimed here.
Native gates, independent C++23 geometry/reference and artifact receipts remain
separate from source implementation.
