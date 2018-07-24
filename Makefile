TEST_PATH=./

all: test clean git

test:
	@echo "Running unit tests."
	@echo ""
	@nosetests -s $(TEST_PATH)
	@echo ""

clean:
	@echo "Cleaning *.pyc, *.DS_Store, and other junk files..."
	@- find . -name '*.pyc' -exec rm --force {} +
	@- find . -name '*.pyo' -exec rm --force {} +
	@echo ""

git: clean
	@echo "Syncing with repository"
	@echo ""
	@- git add --all .
	@- git commit -am "Autocommit from makefile"
	@- git push origin master
