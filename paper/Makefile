# Sounio ArXiv Paper Build

PAPER = sounio-epistemic-types

.PHONY: all clean arxiv

all: $(PAPER).pdf

$(PAPER).pdf: $(PAPER).tex references.bib
	pdflatex $(PAPER)
	bibtex $(PAPER)
	pdflatex $(PAPER)
	pdflatex $(PAPER)

arxiv: $(PAPER).pdf
	mkdir -p arxiv-submission
	cp $(PAPER).tex arxiv-submission/
	cp references.bib arxiv-submission/
	cp $(PAPER).bbl arxiv-submission/
	tar -czvf arxiv-submission.tar.gz arxiv-submission/
	@echo "Created arxiv-submission.tar.gz for arXiv upload"

clean:
	rm -f *.aux *.bbl *.blg *.log *.out *.toc *.pdf
	rm -rf arxiv-submission arxiv-submission.tar.gz
