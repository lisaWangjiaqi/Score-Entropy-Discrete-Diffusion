

import torch

from model import utils as mutils


# =============================================================================
# Conditional Forward Diffusion
# =============================================================================

def corrupt_answer_only(
    graph,
    clean_batch,
    sigma,
    answer_mask,
):
    

    # -------------------------------------------------------------------------
    # First use the original SEDD forward process to corrupt the whole sequence.
    #
    # clean_batch:
    #     """(B, L)"""
    #
    # sigma[:, None]:
    #     """(B,) -> (B, 1)"""
    #
    # noisy_batch:
    #     """(B, L)"""
    # -------------------------------------------------------------------------

    noisy_batch = graph.sample_transition(
        clean_batch,
        sigma[:, None],
    )

    # -------------------------------------------------------------------------
    # Conditional corruption
    #
    # answer_mask == True:
    #     use noisy token
    #
    # answer_mask == False:
    #     restore original clean token
    #
    # Therefore:
    #
    # Prompt  -> clean
    # Answer  -> noisy
    # Padding -> clean
    # -------------------------------------------------------------------------

    perturbed_batch = torch.where(
        answer_mask,
        noisy_batch,
        clean_batch,
    )

    return perturbed_batch


# =============================================================================
# Conditional Score Entropy
# =============================================================================

def get_sft_loss_fn(
    noise,
    graph,
    train=True,
    sampling_eps=1e-3,
):
 

    def loss_fn(
        model,
        batch,
        answer_mask,
        attention_mask=None,
        t=None,
        perturbed_batch=None,
        return_details=False,
    ):
     

        # ---------------------------------------------------------------------
        # Validate inputs
        # ---------------------------------------------------------------------

        if batch.ndim != 2:
            raise ValueError(
                f"batch must have shape (B, L), "
                f"but received {tuple(batch.shape)}"
            )

        if answer_mask.shape != batch.shape:
            raise ValueError(
                f"answer_mask shape {tuple(answer_mask.shape)} "
                f"does not match batch shape {tuple(batch.shape)}"
            )

        # Make sure answer_mask is boolean.
        answer_mask = answer_mask.bool()

        if attention_mask is not None:

            if attention_mask.shape != batch.shape:
                raise ValueError(
                    f"attention_mask shape "
                    f"{tuple(attention_mask.shape)} "
                    f"does not match batch shape "
                    f"{tuple(batch.shape)}"
                )

            attention_mask = attention_mask.bool()

            # Safety:
            # padding positions must never contribute to SFT.
            effective_answer_mask = (
                answer_mask
                & attention_mask
            )

        else:
            effective_answer_mask = answer_mask

        # Every example should contain at least one supervised token.
        answer_token_count = (
            effective_answer_mask
            .sum(dim=-1)
        )

        if torch.any(answer_token_count == 0):
            raise ValueError(
                "At least one example contains no answer tokens."
            )

        # ---------------------------------------------------------------------
        # 1. Sample diffusion time
        #
        # Original SEDD:
        #
        # t ~ Uniform(sampling_eps, 1)
        #
        # Shape:
        #     """(B,)"""
        # ---------------------------------------------------------------------

        if t is None:

            t = (
                (1.0 - sampling_eps)
                * torch.rand(
                    batch.shape[0],
                    device=batch.device,
                )
                + sampling_eps
            )

        else:

            t = t.to(
                device=batch.device,
                dtype=torch.float32,
            )

            if t.ndim != 1:
                t = t.reshape(-1)

            if t.shape[0] != batch.shape[0]:
                raise ValueError(
                    f"t must contain one value per batch item. "
                    f"Received {tuple(t.shape)} for batch "
                    f"{tuple(batch.shape)}."
                )

        # ---------------------------------------------------------------------
        # 2. Convert diffusion time into noise level
        #
        # sigma:
        #     """(B,)"""
        #
        # dsigma:
        #     """(B,)"""
        # ---------------------------------------------------------------------

        sigma, dsigma = noise(t)

        # ---------------------------------------------------------------------
        # 3. Conditional forward diffusion
        #
        # x0:
        #     clean batch
        #
        # xt:
        #     prompt stays clean
        #     answer is corrupted
        #
        # """(B, L) -> (B, L)"""
        # ---------------------------------------------------------------------

        if perturbed_batch is None:

            perturbed_batch = corrupt_answer_only(
                graph=graph,
                clean_batch=batch,
                sigma=sigma,
                answer_mask=effective_answer_mask,
            )

        else:

            perturbed_batch = perturbed_batch.to(
                batch.device
            )

            if perturbed_batch.shape != batch.shape:
                raise ValueError(
                    f"perturbed_batch shape "
                    f"{tuple(perturbed_batch.shape)} "
                    f"does not match batch shape "
                    f"{tuple(batch.shape)}"
                )

            # Important:
            # Even if a custom perturbed_batch is supplied,
            # force prompt/padding positions back to clean values.
            perturbed_batch = torch.where(
                effective_answer_mask,
                perturbed_batch,
                batch,
            )

        # ---------------------------------------------------------------------
        # 4. Get SEDD score function
        #
        # Input:
        #     perturbed_batch: """(B, L)"""
        #     sigma:           """(B,)"""
        #
        # Output:
        #     log_score:       """(B, L, V)"""
        #
        # V = SEDD vocabulary / graph dimension.
        # ---------------------------------------------------------------------

        log_score_fn = mutils.get_score_fn(
            model,
            train=train,
            sampling=False,
        )

        log_score = log_score_fn(
            perturbed_batch,
            sigma,
        )

        # ---------------------------------------------------------------------
        # 5. Original Score Entropy
        #
        # token_loss:
        #     """(B, L)"""
        #
        # This is the same graph-specific Score Entropy used by SEDD.
        # ---------------------------------------------------------------------

        token_loss = graph.score_entropy(
            log_score,
            sigma[:, None],
            perturbed_batch,
            batch,
        )

        # ---------------------------------------------------------------------
        # 6. ANSWER-ONLY LOSS MASK
        #
        # Prompt:
        #     loss -> 0
        #
        # Answer:
        #     keep Score Entropy loss
        #
        # Padding:
        #     loss -> 0
        #
        # """(B, L) -> (B, L)"""
        # ---------------------------------------------------------------------

        masked_token_loss = (
            token_loss
            * effective_answer_mask.to(
                token_loss.dtype
            )
        )

        # ---------------------------------------------------------------------
        # 7. Apply original SEDD diffusion-time weighting
        #
        # Original losses.py:
        #
        # loss = (dsigma[:, None] * loss).sum(dim=-1)
        #
        # We preserve that formulation here.
        #
        # """(B, L) -> (B,)"""
        # ---------------------------------------------------------------------

        weighted_token_loss = (
            dsigma[:, None]
            * masked_token_loss
        )

        loss = weighted_token_loss.sum(
            dim=-1
        )

        # ---------------------------------------------------------------------
        # Optional normalization
        #
        # We deliberately DO NOT divide by the number of answer tokens here,
        # because the original SEDD objective sums over sequence positions.
        #
        # Our arithmetic answers are also very short and similar in length.
        # ---------------------------------------------------------------------

        if not return_details:
            return loss

        # ---------------------------------------------------------------------
        # Debug information
        # ---------------------------------------------------------------------

        details = {
            "t": t.detach(),
            "sigma": sigma.detach(),
            "dsigma": dsigma.detach(),

            "clean_batch": batch.detach(),
            "perturbed_batch": perturbed_batch.detach(),

            "answer_mask": answer_mask.detach(),
            "effective_answer_mask": (
                effective_answer_mask.detach()
            ),

            "log_score": log_score.detach(),

            "token_loss": token_loss.detach(),
            "masked_token_loss": (
                masked_token_loss.detach()
            ),
            "weighted_token_loss": (
                weighted_token_loss.detach()
            ),

            "answer_token_count": (
                answer_token_count.detach()
            ),
        }

        return loss, details

    return loss_fn