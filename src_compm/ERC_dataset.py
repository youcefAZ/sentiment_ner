from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pack_sequence
import random
    
class MELD_loader(Dataset):
    def __init__(self, txt_file, dataclass):
        self.dialogs = []
        
        f = open(txt_file, 'r',encoding="utf8")
        dataset = f.readlines()
        f.close()
        
        temp_speakerList = []
        context = []
        context_speaker = []
        self.speakerNum = []
        # 'anger', 'disgust', 'fear', 'joy', 'neutral', 'sadness', 'surprise'
        emodict = {'anger': "anger", 'disgust': "disgust", 'fear': "fear", 'joy': "joy", 'neutral': "neutral", 'sadness': "sad", 'surprise': 'surprise'}
        self.sentidict = {'positive': ["joy"], 'negative': ["anger", "disgust", "fear", "sadness"], 'neutral': ["neutral", "surprise"]}
        self.emoSet = set()
        self.sentiSet = set()
        for i, data in enumerate(dataset):
            if i < 2:
                continue
            if data == '\n' and len(self.dialogs) > 0:
                self.speakerNum.append(len(temp_speakerList))
                temp_speakerList = []
                context = []
                context_speaker = []
                continue
            speaker, utt, emo, senti = data.strip().split('\t')
            context.append(utt)
            if speaker not in temp_speakerList:
                temp_speakerList.append(speaker)
            speakerCLS = temp_speakerList.index(speaker)
            context_speaker.append(speakerCLS)
            
            self.dialogs.append([context_speaker[:], context[:], emodict[emo], senti])
            self.emoSet.add(emodict[emo])
            self.sentiSet.add(senti)
        
        self.emoList = sorted(self.emoSet)  
        self.sentiList = sorted(self.sentiSet)
        if dataclass == 'emotion':
            self.labelList = self.emoList
        else:
            self.labelList = self.sentiList        
        self.speakerNum.append(len(temp_speakerList))
        
    def __len__(self):
        return len(self.dialogs)

    def __getitem__(self, idx):
        return self.dialogs[idx], self.labelList, self.sentidict
    