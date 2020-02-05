class AssociationRulesExplainer():
    def __init__(self, user_rated_items, association_rules):
        self.user_rated_items = user_rated_items
        self.association_rules = association_rules

    def generate_explanation(self, recommendation):
        rows = self.association_rules.loc[association_rules['consequents']
            .apply(lambda cons: True if recommendation[0] in cons else False)]
        for index, row in rows.iterrows():
            antecedents = list(row['antecedents'])
            if all([x in self.user_rated_items.keys() for x in antecedents]):
                return antecedents  # This is the explanation 

class InfluenceExplainer():
    def __init__(self, user_id, user_rated_items, user_algo):
        self.user_id = user_id
        self.user_rated_items = user_rated_items
        self.user_algo = user_algo

    def generate_explanation(self, recommendation):
        explanations = []
        for i in self.user_rated_items.keys():
            items_copy = self.user_rated_items.copy()
            items_copy.pop(i)

            self.user_algo.delete_user(self.user_id)
            self.user_algo.fit_new_user(self.user_id, items_copy)

            # Test prediction
            prediction = self.user_algo.predict(self.user_id, recommendation[0])
            print("Original: " + str(recommendation[1]) + " Without film: " + str(prediction.est))
            prediction_delta = recommendation[1] - prediction.est
            explanations.append((i, prediction_delta))

        explanations.sort(key=lambda x: x[1], reverse=True)
        positives = explanations[:3]
        negatives = explanations[-3:]
        negatives.reverse()
        return positives, negatives