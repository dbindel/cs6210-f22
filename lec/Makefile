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
	pdf/2022-09-22.pdf \
	pdf/2022-09-27.pdf \
	pdf/2022-09-29.pdf \
	pdf/2022-10-04.pdf \
	pdf/2022-10-06.pdf \
	pdf/2022-10-13.pdf \
	pdf/2022-10-18.pdf \
	pdf/2022-10-20.pdf \
	pdf/2022-10-25.pdf \
	pdf/2022-10-27.pdf \
	pdf/2022-11-01.pdf \
	pdf/2022-11-03.pdf \
	pdf/2022-11-08.pdf \
	pdf/2022-11-10.pdf \
	pdf/2022-11-15.pdf \
	pdf/2022-11-17.pdf \
	pdf/2022-11-29.pdf \
	pdf/2022-12-01.pdf \
	pluto/2022-08-30.jl \
	pluto/2022-08-30.jl.html \
	pluto/2022-09-01.jl \
	pluto/2022-09-01.jl.html \
	pluto/2022-09-08-worksheet.jl \
	pluto/2022-09-08-worksheet.jl.html \
	pluto/2022-09-22.jl \
	pluto/2022-09-22.jl.html

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
