#
mkdir ~/.kaggle
touch ~/.kaggle/kaggle.json
echo '{"username":'\"$KAGGLE_USERNAME\"',"key":'\"$KAGGLE_KEY\"'}' > ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json