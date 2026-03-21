# Sounio Paper Build
# Targets: arxiv (preprint), ieee (CiSE submission)

IEEE     = sounio-ieee-cise
POPL     = popl2027/main
PLDI     = epistemic-types/main
ARXIV_MD = sounio_arxiv_draft.md

.PHONY: all ieee popl pldi arxiv clean

all: ieee

# IEEE CiSE submission
ieee: $(IEEE).pdf

$(IEEE).pdf: $(IEEE).tex references.bib
	pdflatex $(IEEE)
	bibtex $(IEEE)
	pdflatex $(IEEE)
	pdflatex $(IEEE)

# POPL 2027 submission
popl: $(POPL).pdf

$(POPL).pdf: $(POPL).tex popl2027/references.bib
	cd popl2027 && pdflatex main && bibtex main && pdflatex main && pdflatex main

# PLDI 2027 submission
pldi: $(PLDI).pdf

$(PLDI).pdf: $(PLDI).tex references.bib epistemic-types/gpu_benchmarks.tex
	cd epistemic-types && pdflatex main && bibtex main && pdflatex main && pdflatex main

# ArXiv preprint (Markdown source — convert via pandoc)
arxiv: $(ARXIV_MD)
	pandoc $(ARXIV_MD) -o sounio_arxiv_draft.pdf --pdf-engine=pdflatex \
		--variable=geometry:margin=1in --variable=fontsize:11pt || \
		echo "pandoc not available; $(ARXIV_MD) is the canonical arXiv source"

clean:
	rm -f *.aux *.bbl *.blg *.log *.out *.toc *.pdf
	rm -f popl2027/*.aux popl2027/*.bbl popl2027/*.blg popl2027/*.log popl2027/*.out popl2027/*.pdf
	rm -f epistemic-types/*.aux epistemic-types/*.bbl epistemic-types/*.blg epistemic-types/*.log epistemic-types/*.out epistemic-types/*.pdf
