aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin 961104659532.dkr.ecr.us-east-1.amazonaws.com
docker build -t extract-leads-batch . --platform=linux/amd64
docker tag extract-leads-batch:latest 961104659532.dkr.ecr.us-east-1.amazonaws.com/extract-leads-batch:latest
docker push 961104659532.dkr.ecr.us-east-1.amazonaws.com/extract-leads-batch:latest