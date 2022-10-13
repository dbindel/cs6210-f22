TEXINPUTS:=.

HW=	hw1.jl \
	hw2.jl \
	hw3.jl \
	hw4.jl \
	hw5.jl \
	mt.jl \
	hw1.jl.html \
	hw2.jl.html \
	hw3.jl.html \
	hw4.jl.html \
	hw5.jl.html \
	mt.jl.html 

.PHONY: all copy
all: $(HW)

pdf/%.pdf: notes/%.tex
	xelatex -output-dir pdf $<
	xelatex -output-dir pdf $<

clean:
	rm -f pdf/*.aux pdf/*.log pdf/*.out
	rm -f pdf/*.nav pdf/*.snm pdf/*.toc 
	rm -f pdf/*.vrb

distclean: clean
	rm -f pdf/*.pdf

copy:
	mkdir -p ../web/hw
	cp $(HW) ../web/hw
