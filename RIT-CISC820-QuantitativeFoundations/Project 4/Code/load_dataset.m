function train_data = load_dataset(is_train)
% output: 
% image: nxN matrix, each row is an image
% label: Nx1 vector, label for the digit for each image

path = "att_faces/";
if is_train
    train_data = zeros(35 * 8, 92 * 112);
    for i = 1:1:35
        img_folder = strcat(path, "s", int2str(i), "/");
        for j = 3:1:10
            img_file = strcat(img_folder, int2str(j), ".pgm");
            img_read = imread(img_file);
            train_data((i - 1) * 8 + (j - 2),:) = reshape(img_read, 1, []);
        end
    end
else
    test_data = zeros(35 * 2 + 10 * 5, 92 * 112);
    for i = 1:1:35
        img_folder = strcat(path, "s", int2str(i), "/");
        for j = 1:1:2
            img_file = strcat(img_folder, int2str(j), ".pgm");
            img_read = imread(img_file);
            test_data((i - 1) * 2 + j,:) = reshape(img_read, 1, []);
        end
    end
    for i = 36:1:40
        img_folder = strcat(path, "s", int2str(i), "/");
        for j = 1:1:10
            img_file = strcat(img_folder, int2str(j), ".pgm");
            img_read = imread(img_file);
            test_data(35 * 2 + (i - 36) * 10 + j,:) = reshape(img_read, 1, []);
        end
    end
    train_data = test_data;
end
end

    