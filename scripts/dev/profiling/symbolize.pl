#!/usr/bin/perl
# symbolize.pl FNMAP SAMPLES BASE -- flat (self) and inclusive time per function.
# FNMAP lines: "fnmap fn=<i> off=<byte offset in .text> name=<name>"; runtime addr = BASE + off.
use strict; use warnings;
my ($mapf, $sf, $base) = @ARGV; $base = hex($base // '401000');
my (@a, @n);
open(my $m, '<', $mapf) or die $!;
my @rows;
while (<$m>) { next unless /^fnmap fn=\d+ off=(-?\d+) name=(.*)$/; next if $1 < 0; push @rows, [$base + $1, $2]; }
@rows = sort { $a->[0] <=> $b->[0] } @rows; @a = map { $_->[0] } @rows; @n = map { $_->[1] } @rows;
sub sym { my $x = shift; my ($lo,$hi) = (0,$#a); return '?' if !@a || $x < $a[0];
  while ($lo < $hi) { my $mid = int(($lo+$hi+1)/2); if ($a[$mid] <= $x) { $lo = $mid } else { $hi = $mid-1 } } return $n[$lo]; }
my (%self, %incl, $N);
open(my $s, '<', $sf) or die $!;
while (<$s>) { my @f = split; next if @f < 2; $N++; shift @f;
  my @syms = map { sym(hex $_) } @f; $self{$syms[0]}++;
  my %seen; for my $y (@syms) { next if $seen{$y}++; $incl{$y}++ } }
printf "samples=%d functions_mapped=%d\n", $N, scalar @a;
print "== SELF (top 40) ==\n"; my $i=0;
for my $k (sort { $self{$b} <=> $self{$a} } keys %self) { printf "%6.2f%% %7d  %s\n", 100*$self{$k}/$N, $self{$k}, $k; last if ++$i >= 40 }
print "== INCLUSIVE (top 60, depth<=16 frames) ==\n"; $i=0;
for my $k (sort { $incl{$b} <=> $incl{$a} } keys %incl) { printf "%6.2f%% %7d  %s\n", 100*$incl{$k}/$N, $incl{$k}, $k; last if ++$i >= 60 }
