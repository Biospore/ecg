#!/usr/bin/perl

my $folder = shift;

for(my $i = 0; $i <= 20; $i++){
	open my $fh, '<', "$folder/$i.d";
	open my $fho, '>', "$folder/$i.pd";
	while (<$fh>){
		$_ =~ /(\d+\.\d+) (-*\d+\.\d+) (\d)/g;
		print $fho $2."\n";
	}
	close $fh;
	close $fho;
}

# Ольга Эдуардовна
