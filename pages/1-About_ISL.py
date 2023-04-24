import streamlit as st

st.set_page_config(layout="wide")
st.title(':blue[Indian Sign Language(ISL)]')
st.markdown(
    """
    ### Problem Statement:
    The objective of this project is to generate English text from Indian sign language from image, video stream using Deep Neural Network. The generated text can be used in the downstream applications.
    
    ### Background:
    Sign language is used by deaf and hard hearing people to exchange information between their own
    community and with other people. Computer recognition of sign language deals from sign gesture
    acquisition and continues till text/speech generation. Sign gestures can be classified as static and
    dynamic. However static gesture recognition is simpler than dynamic gesture recognition but both
    recognition systems are important to the human community. The key motivation is to explore the
    current research in SRL domain using DNN and generate the real-time text corresponding to the sign
    language.
    Typically, deaf and hard hearing people use sign as their primary language and common spoken
    language such as English, Hindi, etc. are learnt as secondary language. For e.g., a sentence like “My
    name is Mohan” can be said as “Name Mohan is” in sign language.
    Thus, there is a need of interpreters/translators which communicate between both the languages.
    Interpretation and translation can be done with human or with Computers. However, India has only
    around 300 certified interpreters which becomes a supply bottleneck. Interpretation with the help
    of computers is the need of the hour.
    Computers interpretation can be made in terms of isolated and continuous sign language. Isolated
    sign language system is not affected by the previous or following sign. In continuous sign language
    recognition system, different signs are performed one after another to recognize complete sign
    language word or sentence. However, these different signs vary across regions. For e.g. ISL uses two
    hands for communicating (20 out of 26) whereas ASL uses single hand for communicating. Using both
    hands often lead to obscurity of features due to overlapping of hands. These differences make
    complete adoption of the international sign language for Indian Sign Language (ISL) difficult.

    Another approaches of classification of sign language recognition methods can be: (i) Sensor or glove-
    based approach and (ii) Vision based approach. Sensor based methods have advantage of extracting

    the signer’s movements and postures more accurately because they use specialized gloves, which
    are embedded with several sensors to capture the sign information. However, they are difficult to
    deploy in real time. Vision based approach on the other hand, is cost effective and flexible solution
    while touch based approach is complex, costly and difficult to deploy.
    Further, sign language recognition is difficult as the recognition task involves visual and lingual
    information interpretation. Most of the research conducted in Indian sign language recognition has
    considered this task as a gesture recognition problem, and ignores the linguistic properties of the
    sign language and has assumed that there is word-to-word mapping of sign to speech.

    
    ### Motivation:
    Sign language in India is used by almost 63 Lakhs users and with less than 500 certified interpreters
    available, this is a relatively underexplored domain. Moreover, research is primarily focussed on
    recognition of images to predict the signs. Video format is better suited as sign are continuous in
    nature and would help in generating the sentence, thereby leading to clear communication.
    Looking into these limitations, we are motivated to create an application which is able to recognise
    signs and generate English text using both image and video format of Indian Sign Language using
    transfer learning.
    Further, with the help of project, one would be able to apply the concepts learnt throughout the
    course on a real problem statement.

    ### History: 
    
    In the 2000s, the Indian deaf community advocated for an institute focused on ISL teaching and research. The 11th Five Year Plan (2007-2012) acknowledged that the needs of people with hearing disabilities had been relatively neglected and envisaged the development of a sign language research and training center, to promote and develop sign language and training of teachers and interpreters. The Finance Minister announced the setting up of ISLRTC in the Union Budget speech of 2010-11.

    As a result, in 2011, the Ministry of Social Justice and Empowerment approved the establishment of the Indian Sign Language Research and Training Center (ISLRTC) as an autonomous center of the Indira Gandhi National Open University (IGNOU), Delhi. The foundation stone of the center was laid at IGNOU campus on 4th October, 2011. In 2013, the center at IGNOU was closed.

    In an order dated 20th April, 2015, the Ministry decided to integrate ISLRTC with the regional center of the Ali Yavar Jung National Institute of Hearing Handicapped (AYJNIHH) at Delhi. However, the Deaf community protested this decision due to the different perspectives and goals of ISLRTC and AYJNIHH.  

    The protests and meetings with the ministers resulted in the Union Cabinet approving the setting up of ISLRTC as a Society under the Department of Empowerment of Persons with Disabilities, MSJE, in a meeting held on 22nd September, 2015. An order to this effect was issued by the MSJE on 28th September, 2015, leading to the establishment of ISLRTC.

    As per the 2011 Census, the total population of deaf persons in India numbered about 50 lakh.  The needs of the deaf community have long been ignored and the problems have been documented by various organizations working for the deaf. Obsolete training methodology and teaching systems need urgent attention.

    Indian Sign Language (ISL) is used in the deaf community all over India. But ISL is not used in deaf schools to teach deaf children. Teacher training programs do not orient teachers towards teaching methods that use ISL. There is no teaching material that incorporates sign language. Parents of deaf children are not aware about sign language and its ability to remove communication barriers. ISL interpreters are an urgent requirement at institutes and places where communication between deaf and hearing people takes place but India has only less than 300 certified interpreters.

    Therefore, an institute that met all these needs was a necessity.  After a long struggle by the deaf community, the Ministry approved the establishment of ISLRTC in New Delhi on 28th September, 2015. *[Credit](https://islrtc.nic.in/) to ISLRTC 
    
    ### Data Story:
"""
)
st.video('artifacts/datastory/ISL.mp4', 'rb', start_time=0)
st.markdown(
    """    
    ### Use Cases and Future Scope:
    Video conferencing tools, such as Microsoft Teams, Zoom etc, have become a primary source of communication for white-collar employees in the post-pandemic reality. To enable seamless real-time conversation between Deaf and able persons, we will build AI models which that can transcribe signed gestures in real-time. Such real-time transcription can significantly reduce barriers to communicate and enable a wider range of people with no exposure to sign language to effectively communicate with deaf people.

    ### Reference Papers
    - Towards Indian Sign Language Sentence Recognition using INSIGNVID: Indian Sign Language Video Dataset [thesai.org](https://thesai.org/Downloads/Volume12No8/Paper_81-Towards_Indian_Sign_Language_Sentence_Recognition.pdf)
    - https://paperswithcode.com/task/sign-language-recognition

    ### Dataset Links
    - https://data.mendeley.com/datasets/kcmpdxky7p/1
    - https://github.com/DeepKothadiya/Custom_ISLDataset/tree/main
    - https://drive.google.com/drive/folders/1pDMxs6Et6FclAEBVXIdAlAJqA8MboQSf
    - INCLUDE: A Large Scale Dataset for Indian Sign Language Recognition | Zenodo
"""
)