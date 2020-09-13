% Read the phrase and bbox information for all images in Flickr8k into a txt file
fid = fopen('flickr30k_phrases_bboxes.txt', 'w'); 
fid_ptype = fopen('flickr30k_phrase_types.txt', 'w');
fid_sent = fopen('flickr30k_sentences.txt', 'w');
load data_ids.mat
nim = size(data_ids, 2);
ptypes = '';
nptype = 0;

% XXX
for m = 1:nim
    tmp = strsplit(data_ids{m}, '_');
    data_id = tmp{1};
    disp(data_id)
    bb_info = getAnnotations([data_id, '.xml']);
    phrase_info = getSentenceData([data_id, '.txt']);
    phrases = {{}, {}, {}, {}, {}};
    phrase_types = {{}, {}, {}, {}, {}};
    sents = {{}, {}, {}, {}, {}};
    [phrases{1}, phrases{2}, phrases{3}, phrases{4}, phrases{5}] = phrase_info.phrases;
    [phrase_types{1}, phrase_types{2}, phrase_types{3}, phrase_types{4}, phrase_types{5}] = phrase_info.phraseType;
    [sents{1}, sents{2}, sents{3}, sents{4}, sents{5}] = phrase_info.sentence;
    bboxes = [bb_info.labels.boxes];
    nbox = length(bboxes) / 4;
    
    for icap = 1:5
        phrases_i = phrases{icap};
        nph = size(phrases_i, 2);
        fprintf(fid_sent, '%s\n', [data_ids{m}, '_', num2str(icap), ' ', sents{icap}]);
        % Convert phrase cell arr to strings
        for iph = 1:nph
            phrase = phrases_i{iph};
            phrase_type = char(phrase_types{icap}{iph}{1});
            %if strfind(ptypes, phrase_type); else
            fprintf(fid_ptype, '%s\n', [data_ids{m}, '_', num2str(icap), ' ', phrase_type]);
            %    ptypes = [ptypes, phrase_type, ' '];
            %end
            
            nwords = size(phrase, 1);
            pstr = '';
            for iw = 1:nwords
                pstr = [pstr, char(phrase{iw}), ' '];
            end
            % disp(pstr)
            % Cases: If the phrase has a bbox, use the bbox; otherwise the 
            % phrase is a scene, use the entire image as the bbox
            
            if iph < nbox
                phrase_bbox = [pstr, ' ', num2str(bboxes((iph-1)*4+1:iph*4))];
                fprintf(fid, '%s\n', [data_ids{m}, '_', num2str(icap), ' ', phrase_bbox]);
            else
                phrase_bbox = [pstr, ' 0 0 ', num2str(bb_info.dims(1:2))];
                fprintf(fid, '%s\n', [data_ids{m}, '_', num2str(icap), ' ', phrase_bbox]);
            end
        end
    end
end
fclose(fid);
fclose(fid_sent);
fclose(fid_ptype);

