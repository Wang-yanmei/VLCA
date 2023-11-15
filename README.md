# VLCA

## Project Overview

This repo provides code of three key modules for the  paper "Vision and Language Consistency Alignment for Domain Generalization" . The full version of the code will be released upon the acceptance of the paper or its publication on arXiv. Thank you for your attention.


## Three key modules

# inter-class relationship loss based on word-vec
            batch_size = len(label)
            num_classes = len(label_list)
            batch_inter_class_probability = np.zeros((batch_size, num_classes))
            for i, label_num in enumerate(label):
                batch_inter_class_probability[i, :] = probability_distributions[label_list[label_num]]
            batch_inter_class_probability_tensor = torch.from_numpy(batch_inter_class_probability).to(device).float()
            # interclass prob loss*********
            p_scores = F.softmax(scores, dim=1).float()
            loss_interclass = kl_divergence(batch_inter_class_probability_tensor, p_scores)

# domain decouple loss based on CLIP
            feature_text_doamin_ = self.model_clip.encode_text(text_domain.to(device))
            feature_text_doamin = feature_text_doamin_ / feature_text_doamin_.norm(dim=-1, keepdim=True)
            feature_text_class_ = self.model_clip.encode_text(text_class.to(device))  
            feature_text_class = feature_text_class_ / feature_text_class_.norm(dim=-1, keepdim=True)
            cosine_similarity = torch.nn.functional.cosine_similarity(features, feature_text_class, dim=1)
            orthogonal_supervision = torch.abs(
                torch.nn.functional.cosine_similarity(features, feature_text_doamin, dim=1))
            loss_clip = 1 - cosine_similarity.mean() + orthogonal_supervision.mean()
# intraclass  loss based on lowrank decompose
            for l in unique_labels:
                class_features = features[label == l]
                reconstructed_matrix, k = svd_reconstruction(class_features, 0.5)
                pre_loss_proj = k-1
                loss_proj_list.append(pre_loss_proj)
            loss_proj = sum(loss_proj_list) / len(loss_proj_list)
            


