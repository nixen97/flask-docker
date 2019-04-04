MAKEFLAGS += --silent

deploy: build
	docker-compose up -d

run: build
	docker-compose up

build: clean
	docker build -t flask-app:latest .

clean:
	docker-compose rm -f
	
kill:
	docker-compose down
