#
mkdir ~/.kaggle
touch ~/.kaggle/kaggle.json
echo '{"username":'\"$KAGGLE_USERNAME\"',"key":'\"$KAGGLE_KEY\"'}' > ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json
cat ~/.kaggle/kaggle.json

# git hub actions
mkdir /home/runner/.kaggle
touch /home/runner/.kaggle/kaggle.json
echo '{"username":'\"$KAGGLE_USERNAME\"',"key":'\"$KAGGLE_KEY\"'}' > /home/runner/.kaggle/kaggle.json
chmod 600 /home/runner/.kaggle/kaggle.json
cat /home/runner/.kaggle/kaggle.json