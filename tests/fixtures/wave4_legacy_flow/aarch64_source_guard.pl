use strict;use warnings;local $/;
sub read_source {open my $f,'<',$_[0] or die $!;my $s=<$f>;close $f;return $s;}
sub functions {
 my($s)=@_;my %out;
 for my $b(split /(?=^fn )/m,$s) {
  if($b=~/^fn ([A-Za-z0-9_]+)\(/) {
   my $n=$1;
   if($n=~/_a64$/ || $n=~/aarch64/) {die "duplicate AArch64 function $n" if exists$out{$n};$out{$n}=$b;}
  }
 }
 return \%out;
}
my $before=functions(read_source($ARGV[0]));my $after=functions(read_source($ARGV[1]));
die "AArch64 function census changed" if scalar(keys%$before)!=scalar(keys%$after);
for my $n(sort keys%$before) {die "AArch64 function changed $n" if !exists$after->{$n}||$before->{$n} ne $after->{$n};}
print "AARCH64_SOURCE_UNCHANGED functions=",scalar(keys%$before),"\n";
