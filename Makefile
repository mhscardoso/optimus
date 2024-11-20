pres:
	pdflatex \
	-output-directory=presentation \
	presentation/show.tex


cl-pres:
	rm presentation/show.aux \
	presentation/show.log \
	presentation/show.nav \
	presentation/show.out \
	presentation/show.snm \
	presentation/show.toc \
	presentation/show.fls \
	presentation/show.fdb_latexmk \
	presentation/show.synctex.gz
