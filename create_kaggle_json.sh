#
username=$1
key=$2

# typical kaggle dir
mkdir ~/.kaggle
touch ~/.kaggle/kaggle.json
echo '{"username":'\"$username\"',"key":'\"$key\"'}' > ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json

# git hub actions dir
mkdir /home/runner/.kaggle
touch /home/runner/.kaggle/kaggle.json
echo '{"username":'\"$username\"',"key":'\"$key\"'}' > /home/runner/.kaggle/kaggle.json
chmod 600 /home/runner/.kaggle/kaggle.json