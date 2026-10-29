use strict;use warnings;use Digest::SHA qw(sha256_hex);use JSON::PP;local $/;
my($src,$fix,$out)=@ARGV;
sub readf{open my$f,'<',$_[0]or die$!;my$s=<$f>;close$f;return$s;}
my@ledger;
sub extract{my($path,$kind,$name)=@_;my$s=readf("$src/$path");$s=~/^([ \t]*)(?:pub )?\Q$kind\E \Q$name\E(?:\(|\s|\{)/m or die"missing $name";my$at=$-[0];my$indent=$1;my$rest=substr($s,$at);$rest=~/\n\Q$indent\E\}/ or die"end $name";my$body=substr($rest,0,$+[0])."\n";push@ledger,{path=>$path,name=>$name,source_sha256=>sha256_hex($s),body_sha256=>sha256_hex($body),source_line=>1+(substr($s,0,$at)=~tr/\n//)};return$body;}
my$body="use lexer::span::*\nuse parser::ast::*\n";
for my$n(qw(TypeKind)){ $body.=extract('self-hosted/check/types.sio','enum',$n);}
for my$n(qw(TypeEntry TypeEntryList)){$body.=extract('self-hosted/check/types.sio','struct',$n);}
for my$n(qw(FnParamInfo FnParamList)){$body.=extract('self-hosted/check/defs.sio','struct',$n);}
$body.=extract('self-hosted/check/types.sio','fn','ty_i64');
$body.=extract('self-hosted/parser/parser.sio','fn','mk_expr');
$body.=readf("$fix/seams.sio.in");
for my$n(qw(checker_collect_closure_param_list_inplace checker_collect_closure_params_inplace)){$body.=extract('self-hosted/check/check.sio','fn',$n);}
$body.="impl Lowerer{\n";
for my$n(qw(kaxi_sanitize_op_kind kaxi_extract_publish_args)){$body.=extract('self-hosted/ir/lower.sio','fn',$n);}
$body.="}\n";
for my$n(qw(rewrite_hardware_publish_args kaxi_op_kind_from_expr expr_int_lit path_first_is_kaxi exprs_copy_string_list_to_name path_last_name string_list_last_name name_eq3 name_eq4 name_eq8)){$body.=extract('self-hosted/parser/exprs.sio','fn',$n);}
$body.=readf("$fix/main.sio.in");
open my$f,'>',"$out/driver.sio"or die$!;print $f $body;close$f;
open my$l,'>',"$out/EXTRACTIONS.json"or die$!;print $l JSON::PP->new->canonical->pretty->encode({schema=>'sounio.ROOT5.exact-consumers.typed-seams.v1',source_head=>readf("$src/../SOURCE_HEAD"),description=>'Exact three consumer bodies and recursive helper, actual AST/TypeEntry/FnParamList. Checker pointer type-lowering and Lowerer receiver are labeled component seams, not actual compiler storage.',extractions=>\@ledger,seams_sha256=>sha256_hex(readf("$fix/seams.sio.in")),driver_sha256=>sha256_hex($body)});close$l;

