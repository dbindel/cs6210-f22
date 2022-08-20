TEXINPUTS:=.

LEC= \
	pdf/background.pdf \
	pdf/2022-08-23.pdf \
	pdf/2022-08-25.pdf

.PHONY: all copy
all: $(LEC)

pdf/%.pdf: notes/%.tex
	xelatex -output-dir pdf $<
	xelatex -output-dir pdf $<

pdf/2022-08-23.pdf: notes/2022-08-23.tex
	xelatex -output-dir pdf $<
	bibtex pdf/2022-08-23
	xelatex -output-dir pdf $<
	xelatex -output-dir pdf $<

clean:
	rm -f pdf/*.aux pdf/*.log pdf/*.out
	rm -f pdf/*.nav pdf/*.snm pdf/*.toc 
	rm -f pdf/*.vrb

distclean: clean
	rm -f pdf/*.pdf

copy:
	mkdir -p ../web/lec
	cp $(LEC) ../web/lec
