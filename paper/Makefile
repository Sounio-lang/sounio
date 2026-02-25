# Sounio Paper Build
# Targets: arxiv (preprint), ieee (CiSE submission)

PREPRINT = sounio-epistemic-types
IEEE     = sounio-ieee-cise

.PHONY: all ieee arxiv clean

all: ieee

# IEEE CiSE submission
ieee: $(IEEE).pdf

$(IEEE).pdf: $(IEEE).tex references.bib
	pdflatex $(IEEE)
	bibtex $(IEEE)
	pdflatex $(IEEE)
	pdflatex $(IEEE)

# ArXiv preprint
arxiv: $(PREPRINT).pdf

$(PREPRINT).pdf: $(PREPRINT).tex references.bib
	pdflatex $(PREPRINT)
	bibtex $(PREPRINT)
	pdflatex $(PREPRINT)
	pdflatex $(PREPRINT)

arxiv-tar: $(PREPRINT).pdf
	mkdir -p arxiv-submission
	cp $(PREPRINT).tex arxiv-submission/
	cp references.bib arxiv-submission/
	cp $(PREPRINT).bbl arxiv-submission/
	tar -czvf arxiv-submission.tar.gz arxiv-submission/
	@echo "Created arxiv-submission.tar.gz for arXiv upload"

clean:
	rm -f *.aux *.bbl *.blg *.log *.out *.toc *.pdf
	rm -rf arxiv-submission arxiv-submission.tar.gz
