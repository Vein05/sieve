MAIN := main
LATEXMK := $(shell command -v latexmk 2>/dev/null)
PDFLATEX := $(shell command -v pdflatex 2>/dev/null)
TECTONIC := $(shell command -v tectonic 2>/dev/null)

.PHONY: all quick clean watch

all:
ifneq ($(LATEXMK),)
	$(LATEXMK) -pdf -interaction=nonstopmode -halt-on-error $(MAIN).tex
else ifneq ($(PDFLATEX),)
	$(PDFLATEX) -interaction=nonstopmode -halt-on-error $(MAIN).tex
	-bibtex $(MAIN)
	$(PDFLATEX) -interaction=nonstopmode -halt-on-error $(MAIN).tex
	$(PDFLATEX) -interaction=nonstopmode -halt-on-error $(MAIN).tex
else ifneq ($(TECTONIC),)
	$(TECTONIC) $(MAIN).tex
else
	@echo "No TeX engine found (latexmk, pdflatex, or tectonic)." && exit 1
endif

quick:
ifneq ($(PDFLATEX),)
	$(PDFLATEX) -interaction=nonstopmode -halt-on-error $(MAIN).tex
else ifneq ($(TECTONIC),)
	$(TECTONIC) $(MAIN).tex
else
	@echo "No single-pass TeX engine found." && exit 1
endif

watch:
ifneq ($(LATEXMK),)
	$(LATEXMK) -pdf -pvc -interaction=nonstopmode $(MAIN).tex
else
	@echo "watch requires latexmk." && exit 1
endif

clean:
ifneq ($(LATEXMK),)
	$(LATEXMK) -C
endif
	rm -f $(MAIN).aux $(MAIN).bbl $(MAIN).blg $(MAIN).log $(MAIN).out \
	      $(MAIN).fls $(MAIN).fdb_latexmk sections/*.aux
