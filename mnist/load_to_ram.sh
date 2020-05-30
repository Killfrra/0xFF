sudo mount -t tmpfs -o defaults,noatime,size=1024M tmpfs ./ram 
cp -r ../datasets/mini_ru_train ./ram
cp -r ../datasets/mini_ru_test ./ram
