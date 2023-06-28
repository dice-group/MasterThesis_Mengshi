import unittest
from Qald import *


class Test_NER(unittest.TestCase):

    def test_detect_entity_with_flair(self):
        question = Question("de", "Ist Hawaii der Geburtsort von Trump?")
        response = question.send_entity_detection_request("flair_ner")
        self.assertTrue("ent_mentions" in response)

    def test_detect_entity_with_davlan(self):
        question = Question("de", "Ist Hawaii der Geburtsort von Trump?")
        response = question.send_entity_detection_request("davlan_ner")
        self.assertTrue("ent_mentions" in response)

    def test_process_entities(self):
        question = Question("de", "Ist Hawaii der Geburtsort von Trump?")
        ner_response = '''{"components":"flair_ner, mgenre_el","ent_mentions":[{"end":10,"link":"Q782","link_candidates":[["Hawaii","de","Q782"]],"start":4,"surface_form":"Hawaii"},{"end":35,"link":"Q22686","link_candidates":[["Donald Trump","de","Q22686"]],"start":30,"surface_form":"Trump"}],"kb":"wd","lang":"de","placeholder":"00","replace_before":false,"text":"Ist Hawaii der Geburtsort von Trump?"}'''

        entities = question.process_ner_response(ner_response)

        expected = {
            "Hawaii": "Q782",
            "Donald Trump": "Q22686"
        }
        self.assertEqual(entities, expected)

if __name__ == '__main__':
    unittest.main()
