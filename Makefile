MAKEFLAGS += --silent

deploy: build
	docker-compose -d

run: build
	docker-compose up

build: clean
	docker build -t flask-app .

clean:
	docker-compose rm -f
