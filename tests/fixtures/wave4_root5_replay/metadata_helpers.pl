use strict;use warnings;local $/;
sub readfile{open my$f,'<',$_[0]or die$!;my$s=<$f>;close$f;return$s;}
sub function{
 my($s,$name)=@_;
 if($s=~/^(\Q$name\E\(\)\{[^\n]*\})$/m){return "$1\n";}
 $s=~/^(\Q$name\E\(\)\{\n.*?^\})/ms or die"missing $name";
 return "$1\n";
}
my($before,$after,$out)=@ARGV;
for my$arm('before','after'){
 my$s=readfile($arm eq'before'?$before:$after);
 my$body="set -euo pipefail\n".function($s,'sdk_inventory').function($s,'verify_binding');
 if($arm eq'after'){$body.=function($s,'parse_raw_ready');}
 else {
  $s=~/^(  read -r raw_sha raw extra < "\$ready")$/m or die"old read";my$read=$1;
  $s=~/^(  \[\[ "\$raw_sha".*)$/m or die"old validation";my$guard=$1;
  $guard=~s/\|\| die.*$/|| return 1/;
  $body.="parse_raw_ready(){\n local file=\$1 expected=\$2 extra\n local ready=\$file expected_raw=\$expected\n$read\n$guard\n}\n";
 }
 open my$f,'>',"$out/$arm.helpers.sh"or die$!;print $f $body;close$f;
}
