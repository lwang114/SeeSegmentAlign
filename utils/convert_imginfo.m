% Read all the image filenames to a cell array
%{
fn_imfn = '/Users/liming/research/data/flickr/flickr_audio/wav2capt.txt';
fid_imfn = fopen(fn_imfn, 'r');

data_ids = cell(8000, 1);
nim = 1;

while (1)
    tline = fgetl(fid_imfn);
    if (~ischar(tline)) 
        break; 
    end
    
    tline_parts = strsplit(tline);
    if strcmp(tline_parts{3}, '#0')
        tline_parts2 = strsplit(tline_parts{1}, '_');
        imfn = tline_parts{2};
        data_ids{nim} = imfn;
        nim = nim + 1;
    end
    
end
save('data_ids.mat', 'data_ids');
%}
load data_ids.mat
nim = size(data_ids, 2);
bboxes_arr = [];
fid_wrds = fopen('flickr30k_phrases.txt', 'w'); 
for m = 1:nim

tmp = strsplit(data_ids{m}, '_');%'3273892996';
data_id = tmp{1};

bb_info = getAnnotations([data_id, '.xml']);
phrase_info = getSentenceData([data_id, '.txt']);

id = phrase_info.phraseID;
nid = size(id, 2);

ptype = phrase_info.phraseType;
phrases = phrase_info.phrases;
appear_id = '';
id_count = 1;
for k = 1:nid
    % Take the first phrase for each id and ignore the coreferences
    cur_p = phrases{1, k};
    n_w = size(cur_p, 1); 
    cur_pstr = '';
    % Check if the current id already appeared previously, and if so, skip
    % it (because it is coreference); if not, save the id
    cur_id = id{k};
    if strfind(appear_id, cur_id)
        continue
    else
        appear_id = [appear_id, ' ', cur_id];
    end
    
    % Convert the phrase in cell into strings and save the strings along
    % with the phraseid, corresponding bounding boxes in each line of a txt file
    for j = 1:n_w
        cur_pstr = [cur_pstr, cur_p{j}, ' '];
    end

    disp(cur_pstr)
    t_p = ptype{1, k};
    if strcmp(t_p, 'notvisual')
        disp('notvisual')
        continue
    end
    
    % Find the bounding boxes info for the current phrase; if the current
    % phrase contains more than one id, save them separately
    nid = size(bb_info.id, 1);
    iid = 0; % Crash if the current id does not matching any of the id
    for j = 1:nid
        if strcmp(bb_info.id{j}, cur_id)
            iid = j;
            break
        end
    end
    bbids = bb_info.idToLabel(iid);
    bbids_mat = bbids{1};
    nb_curr = size(bbids_mat, 1);
    for l = 1:nb_curr
        % Ignore the scene and nobox annotations
        if isempty(bb_info.labels(bbids_mat(l)).boxes)
            continue
        end
        bboxes_arr = [bboxes_arr; bb_info.labels(bbids_mat(l)).boxes];
        fprintf(fid_wrds, '%s\n', [data_ids{m}, ' ', cur_pstr]);
    end 
end
end
fclose(fid_wrds);
save('bboxes.mat', 'bboxes_arr');
