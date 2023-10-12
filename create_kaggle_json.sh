export KAGGLE_USERNAME=jonathantrull
export KAGGLE_KEY=poopybutthole

touch ~/.kaggle/kaggle.json
echo '{"username":'\"$KAGGLE_USERNAME\"',"key":'\"$KAGGLE_KEY\"'}' > ~/.kaggle/tmp.kaggle.json