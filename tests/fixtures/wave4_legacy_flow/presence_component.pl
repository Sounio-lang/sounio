use strict;use warnings;local $/;
my($old,$new,$main,$out)=@ARGV;
sub source {open my$f,'<',$_[0]or die$!;my$s=<$f>;close$f;return$s;}
my$s=source($new);my$o=source($old);my$m=source($main);
sub func {my($s,$n)=@_;$s=~/^(fn \Q$n\E\(.*?^})/ms or die"missing$n";return"$1\n";}
my$common="var EXPR_VSHADOW: i64 = -1\nvar EXPR_SSHADOW: i64 = -1\nvar EXPR_SSHADOW_1: i64 = -1\nvar EXPR_SSHADOW_2: i64 = -1\nvar EXPR_SSHADOW_3: i64 = -1\nvar EXPR_SSHADOW_4: i64 = -1\nvar EXPR_SSHADOW_5: i64 = -1\nvar EXPR_SSHADOW_6: i64 = -1\nvar EXPR_SSHADOW_7: i64 = -1\nvar EXPR_HSHADOW_00: i64 = -1\nvar EXPR_HSHADOW_01: i64 = -1\nvar EXPR_HSHADOW_02: i64 = -1\nvar EXPR_HSHADOW_03: i64 = -1\nvar EXPR_HSHADOW_04: i64 = -1\nvar EXPR_HSHADOW_05: i64 = -1\nvar EXPR_HSHADOW_06: i64 = -1\nvar EXPR_HSHADOW_07: i64 = -1\nvar EXPR_HSHADOW_11: i64 = -1\nvar EXPR_HSHADOW_12: i64 = -1\nvar EXPR_HSHADOW_13: i64 = -1\nvar EXPR_HSHADOW_14: i64 = -1\nvar EXPR_HSHADOW_15: i64 = -1\nvar EXPR_HSHADOW_16: i64 = -1\nvar EXPR_HSHADOW_17: i64 = -1\nvar EXPR_HSHADOW_22: i64 = -1\nvar EXPR_HSHADOW_23: i64 = -1\nvar EXPR_HSHADOW_24: i64 = -1\nvar EXPR_HSHADOW_25: i64 = -1\nvar EXPR_HSHADOW_26: i64 = -1\nvar EXPR_HSHADOW_27: i64 = -1\nvar EXPR_HSHADOW_33: i64 = -1\nvar EXPR_HSHADOW_34: i64 = -1\nvar EXPR_HSHADOW_35: i64 = -1\nvar EXPR_HSHADOW_36: i64 = -1\nvar EXPR_HSHADOW_37: i64 = -1\nvar EXPR_HSHADOW_44: i64 = -1\nvar EXPR_HSHADOW_45: i64 = -1\nvar EXPR_HSHADOW_46: i64 = -1\nvar EXPR_HSHADOW_47: i64 = -1\nvar EXPR_HSHADOW_55: i64 = -1\nvar EXPR_HSHADOW_56: i64 = -1\nvar EXPR_HSHADOW_57: i64 = -1\nvar EXPR_HSHADOW_66: i64 = -1\nvar EXPR_HSHADOW_67: i64 = -1\nvar EXPR_HSHADOW_77: i64 = -1\nvar EXPR_SX: [i64;24] = [-1;24]\n";
$common.=func($s,'flow_shadow_bit_x86').func($s,'flow_shadow_mask_x86');
for my$a('before','after') {
 my$body=$common.func($a eq'before'?$o:$s,'flow_restore_shadows_x86');
 my$use=$m;
 if($a eq'before') {$use=~s/flow_restore_shadows_x86\(100, m0_([A-Z]+), m1_\1, m2_\1\)/flow_restore_shadows_x86(100)/g;}
 open my$f,'>',"$out/$a.sio"or die$!;print $f $body.$use;close$f;
}
