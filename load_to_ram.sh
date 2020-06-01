sudo mount -t tmpfs -o defaults,noatime,size=1024M tmpfs ./ram 
cp -r datasets/mini_ru_train/no_label ./ram/mini_ru_train/no_label
cp -r datasets/mini_ru_test/no_label ./ram/mini_ru_test/no_label
