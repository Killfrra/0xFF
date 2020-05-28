sudo mount -t tmpfs -o defaults,noatime,size=1024M tmpfs ./ram 
cp -r ../datasets/mini_ru_train_preprocessed ./ram
cp -r ../datasets/mini_ru_test_preprocessed ./ram
