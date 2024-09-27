from __future__ import unicode_literals, print_function, division

print("Beginning AI-model setup...")

from io import open
import unicodedata, re, difflib, torch
import torch.nn as nn
import torch.nn.functional as F
from googletrans import Translator
import random as ra

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


SOS_token = 0
EOS_token = 1
UNK_token = 2

class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {"<UNK>": UNK_token, "<SOS>": SOS_token, "<EOS>": EOS_token}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS", 2: "UNK" }
        self.n_words = 3  # Count SOS and EOS

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1

# Turn a Unicode string to plain ASCII, thanks to
# https://stackoverflow.com/a/518232/2809427
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

# Lowercase, trim, and remove non-letter characters
def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z!?]+", r" ", s)
    return s.strip()

def readLangs(lang1, lang2, reverse=False):
    # Read the file and split into lines
    lines = open('data/%s-%s.txt' % (lang1, lang2), encoding='utf-8').\
        read().strip().split('\n')

    # Split every line into pairs and normalize
    pairs = [[normalizeString(s) for s in l.split('\t')] for l in lines]

    # Reverse pairs, make Lang instances
    if reverse:
        pairs = [list(reversed(p)) for p in pairs]
        input_lang = Lang(lang2)
        output_lang = Lang(lang1)
    else:
        input_lang = Lang(lang1)
        output_lang = Lang(lang2)

    return input_lang, output_lang, pairs

MAX_LENGTH = 100  # Maximum sentence length

eng_prefixes = (
    "i am ", "i m ",
    "he is", "he s ",
    "she is", "she s ",
    "you are", "you re ",
    "we are", "we re ",
    "they are", "they re "
)

def filterPair(p):
    return len(p[0].split(' ')) < MAX_LENGTH and \
        len(p[1].split(' ')) < MAX_LENGTH #and \
        #p[1].startswith(eng_prefixes)


def filterPairs(pairs):
    return [pair for pair in pairs if filterPair(pair)]

def prepareData(lang1, lang2, reverse=False):
    input_lang, output_lang, pairs = readLangs(lang1, lang2, reverse)
    pairs = filterPairs(pairs)
    for pair in pairs:
        input_lang.addSentence(pair[0])
        output_lang.addSentence(pair[1])
    return input_lang, output_lang, pairs


class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, dropout_p=0.1):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, input):
        embedded = self.dropout(self.embedding(input))
        output, hidden = self.gru(embedded)
        return output, hidden

class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(DecoderRNN, self).__init__()
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, encoder_outputs, encoder_hidden, target_tensor=None):
        batch_size = encoder_outputs.size(0)
        decoder_input = torch.empty(batch_size, 1, dtype=torch.long, device=device).fill_(SOS_token)
        decoder_hidden = encoder_hidden
        decoder_outputs = []

        for i in range(MAX_LENGTH):
            decoder_output, decoder_hidden  = self.forward_step(decoder_input, decoder_hidden)
            decoder_outputs.append(decoder_output)

            if target_tensor is not None:
                # Teacher forcing: Feed the target as the next input
                decoder_input = target_tensor[:, i].unsqueeze(1) # Teacher forcing
            else:
                # Without teacher forcing: use its own predictions as the next input
                _, topi = decoder_output.topk(1)
                decoder_input = topi.squeeze(-1).detach()  # detach from history as input

        decoder_outputs = torch.cat(decoder_outputs, dim=1)
        decoder_outputs = F.log_softmax(decoder_outputs, dim=-1)
        return decoder_outputs, decoder_hidden, None # We return `None` for consistency in the training loop

    def forward_step(self, input, hidden):
        output = self.embedding(input)
        output = F.relu(output)
        output, hidden = self.gru(output, hidden)
        output = self.out(output)
        return output, hidden


class BahdanauAttention(nn.Module):
    def __init__(self, hidden_size):
        super(BahdanauAttention, self).__init__()
        self.Wa = nn.Linear(hidden_size, hidden_size)
        self.Ua = nn.Linear(hidden_size, hidden_size)
        self.Va = nn.Linear(hidden_size, 1)

    def forward(self, query, keys):
        scores = self.Va(torch.tanh(self.Wa(query) + self.Ua(keys)))
        scores = scores.squeeze(2).unsqueeze(1)

        weights = F.softmax(scores, dim=-1)
        context = torch.bmm(weights, keys)

        return context, weights

class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, dropout_p=0.1):
        super(AttnDecoderRNN, self).__init__()
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.attention = BahdanauAttention(hidden_size)
        self.gru = nn.GRU(2 * hidden_size, hidden_size, batch_first=True)
        self.out = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, encoder_outputs, encoder_hidden, target_tensor=None):
        batch_size = encoder_outputs.size(0)
        decoder_input = torch.empty(batch_size, 1, dtype=torch.long, device=device).fill_(SOS_token)
        decoder_hidden = encoder_hidden
        decoder_outputs = []
        attentions = []

        for i in range(MAX_LENGTH):
            decoder_output, decoder_hidden, attn_weights = self.forward_step(
                decoder_input, decoder_hidden, encoder_outputs
            )
            decoder_outputs.append(decoder_output)
            attentions.append(attn_weights)

            if target_tensor is not None:
                # Teacher forcing: Feed the target as the next input
                decoder_input = target_tensor[:, i].unsqueeze(1) # Teacher forcing
            else:
                # Without teacher forcing: use its own predictions as the next input
                _, topi = decoder_output.topk(1)
                decoder_input = topi.squeeze(-1).detach()  # detach from history as input

        decoder_outputs = torch.cat(decoder_outputs, dim=1)
        decoder_outputs = F.log_softmax(decoder_outputs, dim=-1)
        attentions = torch.cat(attentions, dim=1)

        return decoder_outputs, decoder_hidden, attentions


    def forward_step(self, input, hidden, encoder_outputs):
        embedded =  self.dropout(self.embedding(input))

        query = hidden.permute(1, 0, 2)
        context, attn_weights = self.attention(query, encoder_outputs)
        input_gru = torch.cat((embedded, context), dim=2)

        output, hidden = self.gru(input_gru, hidden)
        output = self.out(output)

        return output, hidden, attn_weights

def indexesFromSentence(lang, sentence):
    #return [lang.word2index[word] for word in sentence.split(' ')
    indexes = []
    for word in sentence.split(' '):
        if word in lang.word2index:
            indexes.append(lang.word2index[word])
        else:
            indexes.append(lang.word2index['<UNK>'])  # Use a special token for unknown words
    return indexes

def tensorFromSentence(lang, sentence):
    indexes = indexesFromSentence(lang, sentence)
    indexes.append(EOS_token)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(1, -1)

def tensorsFromPair(pair):
    input_tensor = tensorFromSentence(input_lang, pair[0])
    target_tensor = tensorFromSentence(output_lang, pair[1])
    return (input_tensor, target_tensor)

def evaluate(encoder, decoder, sentence, input_lang, output_lang):
    with torch.no_grad():
        input_tensor = tensorFromSentence(input_lang, sentence)

        encoder_outputs, encoder_hidden = encoder(input_tensor)
        decoder_outputs, decoder_hidden, decoder_attn = decoder(encoder_outputs, encoder_hidden)

        _, topi = decoder_outputs.topk(1)
        decoded_ids = topi.squeeze()

        decoded_words = []
        for idx in decoded_ids:
            if idx.item() == EOS_token:
                #decoded_words.append('<EOS>')
                break
            decoded_words.append(output_lang.index2word[idx.item()])
    return decoded_words


input_lang, output_lang, pairs = prepareData('dan', 'eng')
print("Text one loaded...")

input_lang_2, output_lang_2, pairs_2 = prepareData('dan', 'eng2')
print("Text two loaded...")

input_lang_4, output_lang_4, pairs_4 = prepareData('dan', 'eng4')
print("Text three loaded...")

input_lang_5, output_lang_5, pairs_5 = prepareData('dan', 'eng5')
print("All texts loaded... \n")



encoder = EncoderRNN(input_lang.n_words, 600).to(device)
decoder = AttnDecoderRNN(600, output_lang.n_words).to(device)

encoder2 = EncoderRNN(input_lang_2.n_words, 600).to(device)
decoder2 = AttnDecoderRNN(600, output_lang_2.n_words).to(device)

encoder3 = EncoderRNN(input_lang_2.n_words, 200).to(device)
decoder3 = AttnDecoderRNN(200, output_lang_2.n_words).to(device)

encoder4 = EncoderRNN(input_lang_4.n_words, 128).to(device)
decoder4 = AttnDecoderRNN(128, output_lang_4.n_words).to(device)

encoder5 = EncoderRNN(input_lang_5.n_words, 128).to(device)
decoder5 = AttnDecoderRNN(128, output_lang_5.n_words).to(device)

encoder6 = EncoderRNN(input_lang.n_words, 320).to(device)
decoder6 = AttnDecoderRNN(320, output_lang.n_words).to(device)

list_of_encoders = [encoder, encoder2, encoder3, encoder4, encoder5, encoder6]
list_of_decoders = [decoder, decoder2, decoder3, decoder4, decoder5, decoder6]

if torch.cuda.is_available():
    i = 1
    for _encoder, _decoder in zip(list_of_encoders, list_of_decoders):
        a = ""
        b = ""
        if i == 1:
            a = "encoder"
            b = "decoder"
        else:
            a = f"encoderV{i}"
            b = f"decoderV{i}"
        
        _encoder.load_state_dict(torch.load(r'server_side/models/{}.pth'.format(a), 
                                       weights_only=True, 
                                       map_location=torch.device('cuda')))
        _decoder.load_state_dict(torch.load(r'server_side/models/{}.pth'.format(b), 
                                       weights_only=True, 
                                       map_location=torch.device('cuda')))
        
        print(f"Ai_v.{i} Model loaded...")
        i += 1

else:   
    i = 1
    for _encoder, _decoder in zip(list_of_encoders, list_of_decoders):
        a = ""
        b = ""
        if i == 1:
            a = "encoder"
            b = "decoder"
        else:
            a = f"encoderV{i}"
            b = f"decoderV{i}"
        
        _encoder.load_state_dict(torch.load(r'server_side/models/{}.pth'.format(a), 
                                       weights_only=True, 
                                       map_location=torch.device('cpu')))
        _decoder.load_state_dict(torch.load(r'server_side/models/{}.pth'.format(b), 
                                       weights_only=True, 
                                       map_location=torch.device('cpu')))
        
        print(f"Ai_v.{i} Model loaded...")
        i += 1

print("\nAll models loaded... \n")

for _encoder, _decoder in zip(list_of_encoders, list_of_decoders):
    _encoder.eval()
    _decoder.eval()

def clean_output(output_words):
    output_str = ' '.join(output_words) + " "
    
    if output_str != " ":
        output_str = output_str.strip()
    
    output_str = output_str.replace('i m', "i'm")
    output_str = output_str.replace('i m', "she's")
    output_str = output_str.replace('you re', "you're")
    output_str = output_str.replace('we re', "we're")
    output_str = output_str.replace('they re', "they're")
    output_str = output_str.replace('that s', "thats")
    output_str = output_str.replace('it s', "its")
    output_str = output_str.replace('what s', "whats")
    output_str = output_str.replace('where s', "wheres")
    output_str = output_str.replace('who s', "whos")
    output_str = output_str.replace('how s', "hows")
    output_str = output_str.replace('why s', "whys")
    output_str = output_str.replace('SOS', ' ')
    output_str = output_str.replace('EOS', ' ')


    output_str = output_str[0].upper() + output_str[1:] + "."
        
    return output_str

translator = Translator()
def translate(input_sentence, show_ai=True):    
    output_words1 = evaluate(encoder, decoder, normalizeString(str(input_sentence)), input_lang, output_lang)
    output_words2 = evaluate(encoder2, decoder2, normalizeString(str(input_sentence)), input_lang_2, output_lang_2)
    output_words3 = evaluate(encoder3, decoder3, normalizeString(str(input_sentence)), input_lang_2, output_lang_2)
    output_words4 = evaluate(encoder4, decoder4, normalizeString(str(input_sentence)), input_lang_4, output_lang_4)
    output_words5 = evaluate(encoder5, decoder5, normalizeString(str(input_sentence)), input_lang_5, output_lang_5)
    output_words6 = evaluate(encoder6, decoder6, normalizeString(str(input_sentence)), input_lang, output_lang)


 
    output_str1 = clean_output(output_words1)
    output_str2 = clean_output(output_words2)
    output_str3 = clean_output(output_words3)
    output_str4 = clean_output(output_words4)
    output_str5 = clean_output(output_words5)
    output_str6 = clean_output(output_words6)
    
    
    temp = [output_str1, output_str2, output_str3, output_str4, output_str5, output_str6]
    
    translation = translator.translate(normalizeString(str(input_sentence)), dest="en", src="da")
    
    to_clean = []
    
    for i in temp:
        if re.search('[a-zA-Z]', i):
            to_clean.append(i)
        
    
    if not to_clean == []:
        a = find_most_similarV2(translation.text, to_clean)
        b = ""        
        if a == output_str1:
            b = "Ai_v.1"
        elif a == output_str2:
            b = "Ai_v.2"
        elif a == output_str3:
            b = "Ai_v.3"
        elif a == output_str4:
            b = "Ai_v.4"
        elif a == output_str5:
            b = "Ai_v.5"
        elif a == output_str6:
            b = "Ai_v.6"
        else:
            b = "No Ai"
        
        procentage = float(calculate_procentage(translation.text, a))
        
        
        
        if show_ai:
            combined_output = f"{a} ({b}, Accuracy: {procentage:.2f}%)"
        else:
            combined_output = f"{a}"
        
        return combined_output
    else:
        return f"No translation found"
    

def calculate_procentage(head, closest_match):
    head = head.replace(".", "")
    closest_match = closest_match.replace(".", "")
        
    _head = head.lower().split(" ")
    _closest_match = closest_match.lower().split(" ")
    
    max = 0
    matches = 0
    
    def add_match(a, max, mat):
        if a:
            mat += 1
            max += 1
        else:
            max += 1
        return max, mat
    
    if len(_head) == len(_closest_match):
        max, matches = add_match(True, max, matches)
    else:
        max, matches = add_match(False, max, matches)
    
    for i in _head:
        max, matches = add_match(i.strip() in _closest_match, max, matches)
    
    for i in range(len(_head)):
        if i >= len(_closest_match):
            break
        max, matches = add_match(_head[i].strip() == _closest_match[i].strip(), max, matches)
    
    for i in range(len(_head) - 1):
        if (i in _closest_match):
            a = _closest_match.index(i)
            if a == _closest_match[-1]:
                max, matches = add_match(False, max, matches)
                continue
            max, matches = add_match((_head[i].strip() + _head[i + 1].strip()) == (_closest_match[a].strip() + _closest_match[a + 1].strip()), max, matches)
        
    
    return 100 * (matches / max)

def find_most_similar(head, string_list):
    if not string_list:
        return None
    
    # Get the closest match
    closest_matches = difflib.get_close_matches(head, string_list, n=1)
    
    # Return the closest match if found, otherwise None
    return closest_matches[0] if closest_matches else ra.choice(string_list)

def find_most_similarV2(head, string_list):
    if not string_list:
        return None
    
    # Get the closest match
    values = []
    temp = 0
    for i in string_list:
        values.append(calculate_procentage(head, i))
        if max(values) == values[-1]:
            temp = len(values) - 1
    
    closest_matches = string_list[temp]
        
    return closest_matches