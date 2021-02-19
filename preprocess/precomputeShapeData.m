function [] = precomputeShapeData()
    shapenet_dir = './shapenet/';
    savepath = '../datasets/shapenet/';
    mkdir(savepath);
    mkdir([savepath,'/train']);
    mkdir([savepath,'/test']);
    gridSize=32;
    numSamples = 1000; 
    targetnum=4000;
    stepRange = -0.5+1/(2*gridSize):1/gridSize:0.5-1/(2*gridSize);
    [Xp,Yp,Zp] = ndgrid(stepRange, stepRange, stepRange);
    queryPoints = [Xp(:), Yp(:), Zp(:)];

    cates = dir(shapenet_dir);
    for i=3:length(cates)
        fid = fopen(['./data_split/',cates(i).name,'_train.txt'],'r');
        train_ids = textscan(fid,'%s','delimiter','\n'); 
        train_ids = train_ids{1};
        fclose(fid);
        num_per_model = ceil(targetnum / length(train_ids));
        for j=1:size(train_ids,1)

            modelfile = [cates(i).folder,'/',cates(i).name,'/',train_ids{j},'/model.obj'];
            %ignore bad meshes
            try
                [vertices,faces,~]=readOBJ(modelfile);
                [surfaceSamples,~]=meshlpsampling(modelfile,numSamples);
            catch 
                continue;
            end
            if(isempty(vertices) || isempty(faces))
                continue;
            end
            for k=1:num_per_model
                axis = rand(1,3);
                axis = axis/norm(axis);
                angle=rand(1)*2*pi;
                axisangle=[axis,angle];
                R=axang2rotm(axisangle);
                v=R*vertices';
                FV = struct();
                FV.faces=faces;
                FV.vertices = (gridSize)*(v'+0.5) + 0.5;
                Volume=polygon2voxel(FV,gridSize,'none',false);
                sample=R*surfaceSamples;
                [~,~,closestPoints] = point_mesh_squared_distance(queryPoints,v',faces);
                closestPointsGrid = reshape(closestPoints,[size(Xp),3]);
                savefunc([savepath,'/train/',train_ids{j},'_a',num2str(k),'.mat'], Volume,sample, v', faces,axisangle,closestPointsGrid);
            end
        end
        fid = fopen(['./data_split/',cates(i).name,'_test.txt'],'r');
        test_ids = textscan(fid,'%s','delimiter','\n'); 
        test_ids = test_ids{1};
        fclose(fid);
        for j=1:size(test_ids,1)
            modelfile = [cates(i).folder,'/',cates(i).name,'/',test_ids{j},'/model.obj'];
            %ignore bad meshes
            try
                [vertices,faces,~]=readOBJ(modelfile);
                [surfaceSamples,~]=meshlpsampling(modelfile,numSamples);
            catch 
                continue;
            end
            if(isempty(vertices) || isempty(faces))
                continue;
            end
            axis = rand(1,3);
            axis = axis/norm(axis);
            angle=rand(1)*2*pi;
            axisangle=[axis,angle];
            R=axang2rotm(axisangle);
            v=R*vertices';
            FV = struct();
            FV.faces=faces;
            FV.vertices = (gridSize)*(v'+0.5) + 0.5;
            Volume=polygon2voxel(FV,gridSize,'none',false);
            sample=R*surfaceSamples;
            [~,~,closestPoints] = point_mesh_squared_distance(queryPoints,v',faces);
            closestPointsGrid = reshape(closestPoints,[size(Xp),3]);
            savefunc([savepath,'/test/',test_ids{j},'.mat'], Volume,sample, v', faces,axisangle,closestPointsGrid);
        end
    end
end
function savefunc(tsdfFile, Volume, surfaceSamples, vertices, faces,axisangle,closestPoints)
    save(tsdfFile,'Volume', 'surfaceSamples','vertices','faces','axisangle','closestPoints');
end