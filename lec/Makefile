TEXINPUTS:=.

LEC= \
	pdf/background.pdf \
	pdf/2022-08-23.pdf \
	pdf/2022-08-25.pdf \
	pdf/2022-08-30.pdf \
	pdf/2022-09-01.pdf \
	pdf/2022-09-06.pdf \
	pdf/2022-09-08.pdf \
	pdf/2022-09-13.pdf \
	pdf/2022-09-15.pdf \
	pdf/2022-09-20.pdf \
	pluto/2022-08-30.jl \
	pluto/2022-08-30.jl.html \
	pluto/2022-09-01.jl \
	pluto/2022-09-01.jl.html \
	pluto/2022-09-08-worksheet.jl \
	pluto/2022-09-08-worksheet.jl.html

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

pdf/2022-08-30.pdf: notes/2022-08-30.tex
	xelatex -output-dir pdf $<
	bibtex pdf/2022-08-30
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
