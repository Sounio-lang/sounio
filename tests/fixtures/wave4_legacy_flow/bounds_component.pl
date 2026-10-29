use strict;use warnings;local $/;
sub readfile{open my$f,'<',$_[0]or die$!;my$s=<$f>;close$f;return$s;}
sub fnbody{my($s,$n)=@_;if($s=~/^(fn \Q$n\E\([^\n]*\})$/m){return"$1\n";}$s=~/^(fn \Q$n\E\(.*?^})/ms or die"missing$n";return"$1\n";}
my($old,$new,$main,$out)=@ARGV;my$a=readfile($old);my$b=readfile($new);my$m=readfile($main);
my$globals="var SRC: [i8;64] = [0;64]\nvar SRC_LEN:i64=0\nvar TC:i64=0\nvar EN_COUNT:i64=0\nvar ST_COUNT:i64=0\nvar READS:i64=0\nvar BAD_READS:i64=0\nvar BAD_KIND:i64=0\nvar EN: [i64;65536] = [0;65536]\nvar ST: [i64;133120] = [0;133120]\nvar TK: [i64;2097152] = [0;2097152]\nvar TS: [i64;2097152] = [0;2097152]\nvar TE: [i64;2097152] = [0;2097152]\nvar FLOW_EN_DECL_TOK_X86: [i64;256] = [0;256]\nvar FLOW_EN_VALID_X86: [i64;256] = [0;256]\nvar FLOW_EN_VARIANT_TOK_X86: [i64;65536] = [0;65536]\nvar ST_DECL_NS: [i64;2048] = [0;2048]\nvar ST_DECL_NE: [i64;2048] = [0;2048]\n";
my$wrappers="fn get_EN(index:i64) -> i64 with Mut, Panic {\n READS = READS + 1\n if index < 0 || index >= 65536 { if BAD_READS == 0 { BAD_KIND = 1 }; BAD_READS = BAD_READS + 1; return 0 }\n return EN[index as usize]\n}\nfn get_ST(index:i64) -> i64 with Mut, Panic {\n READS = READS + 1\n if index < 0 || index >= 133120 { if BAD_READS == 0 { BAD_KIND = 2 }; BAD_READS = BAD_READS + 1; return 0 }\n return ST[index as usize]\n}\nfn get_TK(index:i64) -> i64 with Mut, Panic {\n READS = READS + 1\n if index < 0 || index >= 2097152 { if BAD_READS == 0 { BAD_KIND = 3 }; BAD_READS = BAD_READS + 1; return 0 }\n return TK[index as usize]\n}\nfn get_TS(index:i64) -> i64 with Mut, Panic {\n READS = READS + 1\n if index < 0 || index >= 2097152 { if BAD_READS == 0 { BAD_KIND = 4 }; BAD_READS = BAD_READS + 1; return 0 }\n return TS[index as usize]\n}\nfn get_TE(index:i64) -> i64 with Mut, Panic {\n READS = READS + 1\n if index < 0 || index >= 2097152 { if BAD_READS == 0 { BAD_KIND = 5 }; BAD_READS = BAD_READS + 1; return 0 }\n return TE[index as usize]\n}\nfn get_FLOW_EN_DECL_TOK_X86(index:i64) -> i64 with Mut, Panic {\n READS = READS + 1\n if index < 0 || index >= 256 { if BAD_READS == 0 { BAD_KIND = 6 }; BAD_READS = BAD_READS + 1; return 0 }\n return FLOW_EN_DECL_TOK_X86[index as usize]\n}\nfn get_FLOW_EN_VALID_X86(index:i64) -> i64 with Mut, Panic {\n READS = READS + 1\n if index < 0 || index >= 256 { if BAD_READS == 0 { BAD_KIND = 7 }; BAD_READS = BAD_READS + 1; return 0 }\n return FLOW_EN_VALID_X86[index as usize]\n}\nfn get_FLOW_EN_VARIANT_TOK_X86(index:i64) -> i64 with Mut, Panic {\n READS = READS + 1\n if index < 0 || index >= 65536 { if BAD_READS == 0 { BAD_KIND = 8 }; BAD_READS = BAD_READS + 1; return 0 }\n return FLOW_EN_VARIANT_TOK_X86[index as usize]\n}\nfn get_ST_DECL_NS(index:i64) -> i64 with Mut, Panic {\n READS = READS + 1\n if index < 0 || index >= 2048 { if BAD_READS == 0 { BAD_KIND = 9 }; BAD_READS = BAD_READS + 1; return 0 }\n return ST_DECL_NS[index as usize]\n}\nfn get_ST_DECL_NE(index:i64) -> i64 with Mut, Panic {\n READS = READS + 1\n if index < 0 || index >= 2048 { if BAD_READS == 0 { BAD_KIND = 10 }; BAD_READS = BAD_READS + 1; return 0 }\n return ST_DECL_NE[index as usize]\n}\n";
for my$arm('before','after'){
 my$s=$arm eq'before'?$a:$b;my$body='';
 for my$n(qw(gl_name_hash name_eq sb flow_delim_matches_x86 flow_validate_enum_x86 flow_enum_for_hash_x86 flow_enum_variant_x86)){ $body.=fnbody($s,$n); }
 $body.=fnbody($b,'flow_counts_valid_x86');
 my$c=($body=~s/\b(EN|ST|TK|TS|TE|FLOW_EN_DECL_TOK_X86|FLOW_EN_VALID_X86|FLOW_EN_VARIANT_TOK_X86|ST_DECL_NS|ST_DECL_NE)\[([^\[\]]+?) as usize\](?!\s*=(?!=))/'get_'.$1.'('.$2.')'/ge);
 die"missing read instrumentation"if$c<20;
 open my$f,'>',"$out/$arm.sio"or die$!;print $f $globals.$wrappers.$body.$m;close$f;
 if($arm eq'before') {
  my @calls=($m=~/^( run_bounds\([^\n]+\))$/mg);
  my $prefix=$m; $prefix=~s/^fn main\(\).*\z//ms;
  for my$i(0..$#calls) {
   my$file=sprintf("$out/before-%03d.sio",$i);
   open my$single,'>',$file or die$!;
   print $single $globals.$wrappers.$body.$prefix."fn main() -> i32 with IO, Mut, Panic, Div {\n".$calls[$i]."\n0\n}\n";
   close$single;
  }
 }
 print "INSTRUMENTED_READS $arm $c\n";
}
