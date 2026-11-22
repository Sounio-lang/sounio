Split the Sporos stage-1 final proof by phase into the chained
modules `Bootstrap.SporosX1FinalDefs`, `SporosX1FinalLoader`,
`SporosX1FinalParser` and `SporosX1FinalRefuse`, registered in the
lakefile so CI compiles everything already proved.
`Bootstrap.SporosX1Final` keeps only the three target theorems.
Declarations were moved verbatim; no statement changed.
