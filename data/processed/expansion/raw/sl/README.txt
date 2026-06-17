The word association norms for Slovenian SWOW-SL 1.0 contain words and their associations collected 
in the scope of the project "Mali svet besed", a Slovenian replication of the experiment 
"Small World of Words" (De Deyne et al. 2019). The SWOW project (https://smallworldofwords.org/en/project) 
is a large-scale scientific study that aims to build a mental dictionary or lexicon by collecting free 
word associations to linguistic cues (words) from human online participants.


SWOW-SL 1.0 contains word associations for 1,000 different cues in Slovenian collected up to November 5, 2024. 
It includes all 19,898 responses collected online from more than 1,100 native Slovenian speakers,
each providing up to 3 associations per given cue. The word association norms - the associative frequency
and associative strength - comprises more than 35,000 unique cue-association pairs.


< Preprocessing >
The responses are provided both in their raw, unprocessed form, and in two normalized forms.
First, the responses were lemmatized using the CLASSLA 2.1 Pipeline (Ljubešić & Dobrovoljc 2019, Terčon & Ljubešič 2023)
for non-standard Slovenian. Using the lemma and named entity (NE) information, we manually checked and amended responses 
to restore diacritics, keep NEs uppercase, and correct some frequent typos. 
A total of 4,387 responses were manually checked, of which 1,078 were changed, including:
	1) responses where lemmatization introduces a diacritic character č/š/ž not present in the original response;
	2) responses with any NEs;
	3) responses in which lemmatization alters the case of the starting letter;
	4) any other one-word response which is not represented in the Slovenian morphological lexicon Sloleks 3.0 (Čibej et al. 2022).
Additionally, where possible, leading and ending punctuation and quotation marks were removed via regular expressions.

< Note on response types >
SWOW-SL 1.0 also allows for 2 special kinds of (non)responses: "<unknownWord>" and "<noMoreReplies>".
- "<unknownWord>" indicates the participant did not know the word 
	and has thus not provided any word associations to it;
- "<noMoreReplies>" indicates the participant stopped responding before 
	reaching the maximum three associations.

< Acknowledgements >
The authors would like to thank the CLARIN.SI consortium for financially supporting
the creation of this resource through the CLARIN.SI project "CLARINprojekt-2024-swow". 


< References >
Čibej, Jaka; et al., 2022, Morphological lexicon Sloleks 3.0, 
	Slovenian language resource repository CLARIN.SI, 
	ISSN 2820-4042, http://hdl.handle.net/11356/1745. 
De Deyne, S., Navarro, D. J., Perfors, A., Brysbaert, M. and Storms, G. 2019.
	The Small World of Words: English word association norms for over 12,000 cue
	words. Behavior Research Methods, 51, 987-1006.
	https://doi.org/10.3758/s13428-018-1115-7
Ljubešić, N., and Dobrovoljc, K. 2019. What does Neural Bring? Analysing Improvements 
	in Morphosyntactic Annotation and Lemmatisation of Slovenian, Croatian and Serbian. 
	Proceedings of the 7th Workshop on Balto-Slavic Natural Language Processing, 29-34. 
	https://doi.org/10.18653/v1/W19-3704.
Terčon, L., and Ljubešić, N. 2023. CLASSLA-Stanza: The Next Step for Linguistic Processing 
	of South Slavic Languages. arXiv preprint. 
	https://arxiv.org/abs/2308.04255



< Files >
SWOW-SL 1.0 is comprised of 4 files in tabular .tsv format, namely 
	#1 SWOW-SL1.0_responses.tsv, which contains all collected responses,
	#2 SWOW-SL1.0_participants.tsv, which contains participant metadata,
	#3 SWOW-SL1.0_statistics_raw.tsv, which contains the word association norms, i.e. cue-association frequency statistics 
		calculated on the basis of original, unprocessed responses.
	#4 SWOW-SL1.0_statistics_normalized.tsv, which contains the word association norms, i.e. cue-association frequency statistics 
		calculated on the basis of normalized responses.
	#5 manual_changes, which contains mappings for all manually corrected responses.
The description of the file contents are given below.

#1 SWOW-SL1.0_responses.tsv #

Contains all responses given by participants, delimited into columns:
 - id				ID of a complete response by a participant
 - participantID	Participant ID in integer format (*matches 'participantID' in SWOW-SL_participants.tsv')
 - cue				The cueword presented to the participant
 - response1		The participant's 1st association to the cue
 - response2		The participant's 2nd association to the cue
 - response3		The participant's 3rd association to the cue

The file also contains two types of normalized responses:
a) Lemmatized responses: the word-lemmatized form of each response using the 
	non-standard language version of the CLASSLA Pipeline 2.1; columns
 - response1_lemmas
 - response2_lemmas
 - response3_lemmas
b) Normalized responses; columns
 - response1Normalized
 - response2Normalized
 - response3Normalized

# 2 SWOW-SL1.0_participants.tsv #
Contains participant metadata, including:
 - participantID	Participant ID in integer format (*corresponds to 'participantID' in SWOW-SL_responses.tsv')
 - age				Age of the participant (ranges from 18 to 100)
 - education		Level of education attained by the participant, possible values:
						1	drugo [other]
						2	osnovna šola [elementary school]
						3	srednja šola [high school]
                        4	višja ali strokovna šola [professional diploma]                                
						5	visokošolska ali univerzitetna diploma 1. stopnje. [undergraduate/Bachelor's]
                        6	visokošolska ali univerzitetna diploma 2. stopnje. [graduate/Master's]
                        7	doktorat znanosti [postgraduate/doctorate]
 - gender			Gender of the participant, possible values:
						Fe	ženski [Female]
						Ma	moški [Male]
						X	drugo/ne želim povedati [Other/rather not say]
 
 
3# SWOW-SL1.0_statistics_raw.tsv and 
4# SWOW-SL1.0_statistics_normalized.tsv
These two files contain some basic statistics, i.e. the frequency and association strength of associations for a given cue.

SWOW-SL1.0_statistics_raw.tsv contains cue-association statistics calculated on the basis of 
	the raw, unprocessed format (response1, response2, response3 in 'SWOW-SL_responses.tsv'),
	including non-responses "<noMoreReplies>" and "<unknownWord>".
SWOW-SL1.0_statistics_normalized.tsv contains cue-association statistics calculated on the basis of 
	the normalized format (response1Normalized, response2Normalized, response3Normalized in 'SWOW-SL_responses.tsv').

The files contain the following columns:
 - cue					The word appearing as the cue
 - response				The word(s) given as an association to the cue
 - isCue				Indicates whether the association is also in the list of cues in SWOW-SL
 - F1					Frequency of the association as the 1st response to the cue
 - F2					Frequency of the association as the 2nd response to the cue
 - F3					Frequency of the association as the 3rd response to the cue
 - F123					The total frequency of an association given as a response to the cue (F1 + F2 + F3)
 - association_strength	The association strength of a particular association relative to the cue, 
							calculated as the conditional probability of the association given 
							all responses to the cue:  P(A | B) = P(A ∩ B) / P(B)	

