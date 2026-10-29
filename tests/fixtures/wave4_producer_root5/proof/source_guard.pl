use strict;use warnings;use Digest::SHA qw(sha256_hex);local $/;
sub readf{open my$f,'<',$_[0]or die$!;my$s=<$f>;close$f;return$s;}
sub body{my($s,$n)=@_;$s=~/^(fn \Q$n\E\(.*?^})/ms or die$n;return$1;}
my($a,$b)=map{readf($_)}@ARGV;
for my$n(qw(ty_eq tc_mark_failed tc_error)){die"changed $n"if body($a,$n)ne body($b,$n);print"UNCHANGED $n ".sha256_hex(body($b,$n))."\n";}

