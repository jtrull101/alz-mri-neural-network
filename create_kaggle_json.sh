#
username=$1
key=$2

# typical kaggle dir
mkdir ~/.kaggle
touch ~/.kaggle/kaggle.json
echo '{"username":'\"$username\"',"key":'\"$key\"'}' > ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json

# # typical kaggle dir
# mkdir /root/.kaggle
# touch /root/.kaggle/kaggle.json
# echo '{"username":'\"$username\"',"key":'\"$key\"'}' > /root/.kaggle/kaggle.json
# chmod 600 /root/.kaggle/kaggle.json

# git hub actions dir
mkdir /home/runner/.kaggle
touch /home/runner/.kaggle/kaggle.json
echo '{"username":'\"$username\"',"key":'\"$key\"'}' > /home/runner/.kaggle/kaggle.json
chmod 600 /home/runner/.kaggle/kaggle.json